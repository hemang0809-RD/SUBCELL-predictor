"""Step 5 — Predict subcellular localization for new protein sequences.

Usage examples:
    # From a FASTA file:
    python predict.py --fasta query.fasta

    # From a raw sequence on the command line:
    python predict.py --seq MSTIKLGLLVLFVAQFALLSQAGSTAEGKLIVEDD

    # Specify a custom model directory:
    python predict.py --seq ACDEFGHIKLMNPQRSTVWY --model-dir models/
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from subcell_predictor.features import build_feature_matrix

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"


def load_model(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> tuple:
    """Load trained model, label encoder, and feature metadata.

    Returns
    -------
    clf : RandomForestClassifier
    le : LabelEncoder
    meta : dict
    """
    model_dir = Path(model_dir)

    model_path = model_dir / "rf_model.pkl"
    le_path = model_dir / "label_encoder.pkl"
    meta_path = model_dir / "model_meta.json"

    for p in (model_path, le_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing artifact: {p}\n"
                "Run 'python -m subcell_predictor.train --fasta <path>' first."
            )

    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    meta = json.loads(meta_path.read_text())

    return clf, le, meta


def predict_sequences(
    sequences: list[str],
    names: list[str] | None = None,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> pd.DataFrame:
    """Predict subcellular localization for a list of sequences.

    Parameters
    ----------
    sequences : list[str]
        Raw amino-acid strings.
    names : list[str] | None
        Optional identifiers for each sequence.
    model_dir : str | Path
        Directory containing model artifacts.

    Returns
    -------
    pd.DataFrame
        Columns: name, sequence (truncated), predicted_location, confidence,
        and per-class probabilities.
    """
    clf, le, meta = load_model(model_dir)
    feature_columns: list[str] = meta["feature_columns"]

    # Detect which feature sets were used during training.
    has_aac = any(c.startswith("AAC_") for c in feature_columns)
    has_dpc = any(c.startswith("DPC_") for c in feature_columns)
    has_kmer = any(c.startswith("KM_") for c in feature_columns)
    has_phys = any(c.startswith("PC_") for c in feature_columns)

    X = build_feature_matrix(
        sequences,
        use_aac=has_aac,
        use_dpc=has_dpc,
        use_kmer=has_kmer,
        use_physicochemical=has_phys,
    )

    # Ensure column alignment with training data.
    missing = set(feature_columns) - set(X.columns)
    if missing:
        for col in missing:
            X[col] = 0.0
    X = X[feature_columns]

    proba = clf.predict_proba(X)
    pred_indices = np.argmax(proba, axis=1)
    pred_labels = le.inverse_transform(pred_indices)
    confidences = np.max(proba, axis=1)

    if names is None:
        names = [f"seq_{i+1}" for i in range(len(sequences))]

    results = pd.DataFrame(
        {
            "name": names,
            "sequence": [s[:40] + "…" if len(s) > 40 else s for s in sequences],
            "predicted_location": pred_labels,
            "confidence": np.round(confidences, 4),
        }
    )

    # Append per-class probabilities.
    for i, cls_name in enumerate(le.classes_):
        results[f"prob_{cls_name}"] = np.round(proba[:, i], 4)

    return results


def _read_fasta_sequences(fasta_path: str | Path) -> tuple[list[str], list[str]]:
    """Read sequences and IDs from a FASTA file."""
    names, seqs = [], []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        names.append(record.id)
        seqs.append(str(record.seq))
    return seqs, names


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Predict subcellular localization of protein sequences"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fasta", help="Path to a FASTA file with query sequences")
    group.add_argument("--seq", help="A single amino-acid sequence string")
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing model artifacts",
    )
    args = parser.parse_args(argv)

    if args.seq:
        sequences = [args.seq.upper()]
        names = ["query"]
    else:
        sequences, names = _read_fasta_sequences(args.fasta)

    if not sequences:
        print("No sequences provided.", file=sys.stderr)
        sys.exit(1)

    results = predict_sequences(sequences, names=names, model_dir=args.model_dir)

    print()
    print(results.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
