"""
src/embedder.py
===============
ESM-2 protein sequence embedder.

Produces mean-pooled last-layer representations for protein sequences.
Handles batching, GPU/CPU device selection, and disk caching.

Usage:
    from src.embedder import ProteinEmbedder

    embedder = ProteinEmbedder()
    E = embedder.embed(["ACDEFGHIKLMNPQRSTVWY", "CNCKAPETALCARRCQQH"])
    # E.shape == (2, 640)

    # With caching
    embedder = ProteinEmbedder(cache_dir="poc_cache")
    E = embedder.embed_with_cache(sequences, cache_key="my_dataset_120")
"""

import numpy as np
import torch
import time
from pathlib import Path
from typing import Optional


# Available ESM-2 checkpoints (smallest → largest)
ESM2_MODELS = {
    "8M":   "facebook/esm2_t6_8M_UR50D",
    "35M":  "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
}
DEFAULT_MODEL = "150M"


class ProteinEmbedder:
    """
    ESM-2 protein sequence embedder.

    Parameters
    ----------
    model_size : str
        ESM-2 model size: "8M", "35M", "150M", or "650M".
    max_length : int
        Maximum sequence length in amino acids (longer sequences truncated).
    batch_size : int
        Batch size for inference. Reduce if OOM.
    cache_dir : str or Path, optional
        Directory for embedding caches. If None, caching is disabled.
    device : str, optional
        "cuda", "cpu", or None (auto-detect).
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        max_length: int = 500,
        batch_size: int = 8,
        cache_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        self.model_size = model_size
        self.model_name = ESM2_MODELS[model_size]
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir  = Path(cache_dir) if cache_dir else None
        self._model     = None
        self._tokenizer = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModel
        print(f"Loading ESM-2 ({self.model_size}) on {self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModel.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )
        free_gb = torch.cuda.mem_get_info()[0] / 1e9 if self.device == "cuda" else 0
        print(f"  Model loaded. GPU free: {free_gb:.1f}GB")

    def embed(self, sequences: list[str], verbose: bool = True) -> np.ndarray:
        """
        Embed a list of protein sequences.

        Parameters
        ----------
        sequences : list of str
            Amino acid sequences (uppercase, standard alphabet).
        verbose : bool
            Print progress.

        Returns
        -------
        np.ndarray, shape (N, D)
            Mean-pooled last-layer embeddings.
        """
        self._load_model()
        all_embeddings = []
        n = len(sequences)
        t0 = time.time()

        for i in range(0, n, self.batch_size):
            batch = sequences[i : i + self.batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length + 2,  # +2 for special tokens
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Mean-pool over residue dimension, excluding special tokens
            hidden = outputs.last_hidden_state          # (B, L, D)
            mask   = inputs["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)            # (B, D)

            all_embeddings.append(pooled.cpu().numpy())

            if verbose and (i // self.batch_size + 1) % 10 == 0:
                done = min(i + self.batch_size, n)
                elapsed = time.time() - t0
                rate = done / elapsed
                eta  = (n - done) / rate / 60
                print(f"  [{done/n:.1%}] {done}/{n} | {rate:.0f} seqs/s | ETA {eta:.1f}min",
                      end="\r", flush=True)

        if verbose:
            elapsed = time.time() - t0
            print(f"\n  Embedded {n} sequences in {elapsed/60:.1f}min "
                  f"({elapsed/n:.3f}s/seq)")

        return np.vstack(all_embeddings)

    def embed_with_cache(
        self,
        sequences: list[str],
        cache_key: str,
    ) -> np.ndarray:
        """
        Embed sequences with disk caching.

        Cache is keyed by cache_key. If a cache file exists with the
        correct shape, it is loaded directly.

        Parameters
        ----------
        sequences : list of str
        cache_key : str
            Unique identifier for this set of sequences.

        Returns
        -------
        np.ndarray
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir must be set to use embed_with_cache")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        model_short = self.model_name.split("/")[-1]
        cache_path = self.cache_dir / f"embeddings_{model_short}_{cache_key}.npy"

        if cache_path.exists():
            E = np.load(cache_path)
            if E.shape[0] == len(sequences):
                print(f"  [cache] Loaded {E.shape} from {cache_path.name}")
                return E
            else:
                print(f"  [cache] Shape mismatch ({E.shape[0]} vs {len(sequences)}), re-embedding")

        E = self.embed(sequences)
        np.save(cache_path, E)
        print(f"  [cache] Saved {E.shape} to {cache_path.name}")
        return E

    def unload(self):
        """Free GPU memory by unloading the model."""
        if self._model is not None:
            del self._model, self._tokenizer
            self._model = self._tokenizer = None
            if self.device == "cuda":
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            print("ESM-2 unloaded.")

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension for the loaded model."""
        dims = {"8M": 320, "35M": 480, "150M": 640, "650M": 1280}
        return dims[self.model_size]
