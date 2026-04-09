"""Featurization module — converts raw amino-acid sequences into numerical feature vectors.

Feature sets:
1. **Amino Acid Composition (AAC)** — percentage of each of the 20 standard amino acids.
2. **Dipeptide Composition (DPC)** — frequency of all 2-mer (dipeptide) combinations (400 features).
3. **K-mer (tri-peptide) counting** — frequency of all 3-mer combinations (8 000 features).
4. **Physicochemical properties** — molecular weight, charge, hydrophobicity, aromaticity, etc.
5. **Sequence length** — raw length of the protein sequence.
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd

# The 20 standard amino acids (single-letter codes).
AMINO_ACIDS: list[str] = sorted(list("ACDEFGHIKLMNPQRSTVWY"))

# Pre-compute all 3-mers for consistent column ordering (20^3 = 8 000).
TRIMERS: list[str] = ["".join(t) for t in product(AMINO_ACIDS, repeat=3)]

# ---------------------------------------------------------------------------
# Physicochemical property tables (per amino acid)
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity scale
_HYDROPHOBICITY: dict[str, float] = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Amino acid molecular weights (Da)
_MOLECULAR_WEIGHT: dict[str, float] = {
    "A": 89.1, "C": 121.2, "D": 133.1, "E": 147.1, "F": 165.2,
    "G": 75.0, "H": 155.2, "I": 131.2, "K": 146.2, "L": 131.2,
    "M": 149.2, "N": 132.1, "P": 115.1, "Q": 146.2, "R": 174.2,
    "S": 105.1, "T": 119.1, "V": 117.1, "W": 204.2, "Y": 181.2,
}

# Charge at pH 7
_CHARGE: dict[str, float] = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0,
    "G": 0, "H": 0.1, "I": 0, "K": 1, "L": 0,
    "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1,
    "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0,
}

_AROMATIC = set("FWY")
_POLAR = set("STNQCY")
_TINY = set("AGCS")


# ---------------------------------------------------------------------------
# Amino Acid Composition (AAC)
# ---------------------------------------------------------------------------

def amino_acid_composition(sequence: str) -> dict[str, float]:
    """Return the percentage composition of each standard amino acid.

    Non-standard residues (B, X, Z, etc.) are ignored in the denominator so
    that the 20 percentages still sum to ~100 for clean sequences.
    """
    seq_upper = sequence.upper()
    counts = {aa: 0 for aa in AMINO_ACIDS}
    total = 0
    for residue in seq_upper:
        if residue in counts:
            counts[residue] += 1
            total += 1

    if total == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}

    return {aa: (count / total) * 100.0 for aa, count in counts.items()}


def compute_aac_matrix(sequences: Sequence[str]) -> pd.DataFrame:
    """Vectorised AAC computation for a collection of sequences.

    Returns a DataFrame of shape ``(n_sequences, 20)`` with column names
    ``AAC_A``, ``AAC_C``, ... , ``AAC_Y``.
    """
    rows = [amino_acid_composition(seq) for seq in sequences]
    df = pd.DataFrame(rows, columns=AMINO_ACIDS)
    df.columns = [f"AAC_{aa}" for aa in AMINO_ACIDS]
    return df


# ---------------------------------------------------------------------------
# Dipeptide Composition (DPC)
# ---------------------------------------------------------------------------

def dipeptide_composition(sequence: str) -> dict[str, float]:
    """Return the frequency of each dipeptide (2-mer) as a fraction."""
    seq_upper = sequence.upper()
    counts: dict[str, int] = {}
    total = 0

    for i in range(len(seq_upper) - 1):
        dp = seq_upper[i : i + 2]
        if dp[0] in _HYDROPHOBICITY and dp[1] in _HYDROPHOBICITY:
            counts[dp] = counts.get(dp, 0) + 1
            total += 1

    if total == 0:
        return {}
    return {dp: count / total for dp, count in counts.items()}


def compute_dpc_matrix(sequences: Sequence[str]) -> pd.DataFrame:
    """Build a dipeptide frequency matrix (n_sequences x 400)."""
    all_dipeptides = ["".join(t) for t in product(AMINO_ACIDS, repeat=2)]
    rows: list[dict[str, float]] = []

    for seq in sequences:
        freq = dipeptide_composition(seq)
        rows.append({dp: freq.get(dp, 0.0) for dp in all_dipeptides})

    df = pd.DataFrame(rows, columns=all_dipeptides)
    df.columns = [f"DPC_{dp}" for dp in all_dipeptides]
    return df


# ---------------------------------------------------------------------------
# K-mer (tri-peptide) frequency
# ---------------------------------------------------------------------------

def kmer_frequencies(sequence: str, k: int = 3) -> dict[str, float]:
    """Count the frequency (as a fraction) of each k-mer in *sequence*.

    Only k-mers composed entirely of standard amino acids are counted.
    """
    seq_upper = sequence.upper()
    total = 0
    counts: dict[str, int] = {}

    for i in range(len(seq_upper) - k + 1):
        kmer = seq_upper[i : i + k]
        # Skip windows containing non-standard residues.
        if all(c in AMINO_ACIDS for c in kmer):
            counts[kmer] = counts.get(kmer, 0) + 1
            total += 1

    if total == 0:
        return {}

    return {kmer: count / total for kmer, count in counts.items()}


def compute_kmer_matrix(
    sequences: Sequence[str],
    k: int = 3,
) -> pd.DataFrame:
    """Build a k-mer frequency matrix for a collection of sequences.

    Returns a DataFrame of shape ``(n_sequences, 20^k)`` with consistent
    column ordering regardless of which k-mers actually appear.
    """
    all_kmers = ["".join(t) for t in product(AMINO_ACIDS, repeat=k)]
    rows: list[dict[str, float]] = []

    for seq in sequences:
        freq = kmer_frequencies(seq, k=k)
        rows.append({km: freq.get(km, 0.0) for km in all_kmers})

    df = pd.DataFrame(rows, columns=all_kmers)
    df.columns = [f"KM_{km}" for km in all_kmers]
    return df


# ---------------------------------------------------------------------------
# Physicochemical properties
# ---------------------------------------------------------------------------

def physicochemical_features(sequence: str) -> dict[str, float]:
    """Compute global physicochemical descriptors for a protein sequence."""
    seq_upper = sequence.upper()
    std_residues = [r for r in seq_upper if r in _HYDROPHOBICITY]
    n = len(std_residues)

    if n == 0:
        return {
            "PC_length": 0, "PC_log_length": 0,
            "PC_mw": 0, "PC_avg_hydrophobicity": 0, "PC_net_charge": 0,
            "PC_frac_aromatic": 0, "PC_frac_polar": 0, "PC_frac_tiny": 0,
            "PC_frac_positive": 0, "PC_frac_negative": 0,
            "PC_gravy": 0, "PC_abs_charge": 0,
        }

    hydro_values = [_HYDROPHOBICITY[r] for r in std_residues]
    charges = [_CHARGE[r] for r in std_residues]

    n_positive = sum(1 for r in std_residues if r in "KR")
    n_negative = sum(1 for r in std_residues if r in "DE")
    n_aromatic = sum(1 for r in std_residues if r in _AROMATIC)
    n_polar = sum(1 for r in std_residues if r in _POLAR)
    n_tiny = sum(1 for r in std_residues if r in _TINY)

    total_mw = sum(_MOLECULAR_WEIGHT[r] for r in std_residues) - (n - 1) * 18.015

    return {
        "PC_length": n,
        "PC_log_length": np.log1p(n),
        "PC_mw": total_mw,
        "PC_avg_hydrophobicity": np.mean(hydro_values),
        "PC_net_charge": sum(charges),
        "PC_frac_aromatic": n_aromatic / n,
        "PC_frac_polar": n_polar / n,
        "PC_frac_tiny": n_tiny / n,
        "PC_frac_positive": n_positive / n,
        "PC_frac_negative": n_negative / n,
        "PC_gravy": sum(hydro_values) / n,
        "PC_abs_charge": abs(sum(charges)),
    }


def compute_physicochemical_matrix(sequences: Sequence[str]) -> pd.DataFrame:
    """Compute physicochemical features for a collection of sequences."""
    rows = [physicochemical_features(seq) for seq in sequences]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Combined feature builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    sequences: Sequence[str],
    use_aac: bool = True,
    use_dpc: bool = False,
    use_kmer: bool = True,
    use_physicochemical: bool = False,
    k: int = 3,
) -> pd.DataFrame:
    """Build a combined feature matrix from selected feature sets.

    Parameters
    ----------
    sequences:
        Iterable of protein sequences (amino-acid strings).
    use_aac:
        Include amino-acid composition features (20 cols).
    use_dpc:
        Include dipeptide composition features (400 cols).
    use_kmer:
        Include k-mer frequency features (20^k cols).
    use_physicochemical:
        Include physicochemical property features (12 cols).
    k:
        K-mer length (default 3).

    Returns
    -------
    pd.DataFrame
        Combined feature matrix ready for ML.
    """
    parts: list[pd.DataFrame] = []

    if use_aac:
        parts.append(compute_aac_matrix(sequences))
    if use_dpc:
        parts.append(compute_dpc_matrix(sequences))
    if use_kmer:
        parts.append(compute_kmer_matrix(sequences, k=k))
    if use_physicochemical:
        parts.append(compute_physicochemical_matrix(sequences))

    if not parts:
        raise ValueError("At least one feature set must be enabled.")

    return pd.concat(parts, axis=1)
