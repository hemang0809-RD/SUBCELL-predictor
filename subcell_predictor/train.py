"""Classification pipeline: train Random Forest or Gradient Boosting with optional SMOTE.

Usage:
    python -m subcell_predictor.train --tsv data/annotated_proteins.tsv
    python -m subcell_predictor.train --tsv data/annotated_proteins.tsv --classifier gb --smote
    python -m subcell_predictor.train --tsv data/annotated_proteins.tsv --aac-only
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from subcell_predictor.features import build_feature_matrix
from subcell_predictor.parser import parse_fasta, parse_tsv, records_to_dataframe

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2


def prepare_data(
    fasta_path: str | Path | None = None,
    tsv_path: str | Path | None = None,
    use_aac: bool = True,
    use_dpc: bool = False,
    use_kmer: bool = True,
    use_physicochemical: bool = False,
    k: int = 3,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Parse FASTA or TSV, featurize sequences, and encode labels.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Integer-encoded labels.
    le : LabelEncoder
        Fitted label encoder (for inverse transform later).
    """
    if tsv_path:
        records = parse_tsv(tsv_path)
    elif fasta_path:
        records = parse_fasta(fasta_path)
    else:
        raise ValueError("Provide either --fasta or --tsv.")

    if not records:
        raise ValueError(
            "No valid records found. "
            "Ensure the file contains subcellular location annotations "
            "for Nucleus, Cytoplasm, or Mitochondrion."
        )

    df = records_to_dataframe(records)
    print(f"Parsed {len(df)} proteins:")
    print(df["location"].value_counts().to_string())
    print()

    sequences = df["sequence"].tolist()
    X = build_feature_matrix(
        sequences,
        use_aac=use_aac,
        use_dpc=use_dpc,
        use_kmer=use_kmer,
        use_physicochemical=use_physicochemical,
        k=k,
    )

    le = LabelEncoder()
    y = le.fit_transform(df["location"].values)

    return X, y, le


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    classifier: str = "rf",
    use_smote: bool = False,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = 200,
    n_jobs: int = -1,
) -> tuple:
    """Train a classifier and return the model + train/test splits.

    Parameters
    ----------
    classifier : str
        "rf" for Random Forest, "gb" for Gradient Boosting.
    use_smote : bool
        Apply SMOTE oversampling to the training set to handle class imbalance.

    Returns
    -------
    clf, X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- SMOTE oversampling on training data only ---
    if use_smote:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=random_state)
        X_train_arr, y_train = smote.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        print(f"SMOTE applied — training set resampled to {len(y_train)} samples")
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c}")
        print()

    # --- Build classifier ---
    if classifier == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
        )
        clf_name = f"Gradient Boosting ({n_estimators} trees)"
    else:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        clf_name = f"Random Forest ({n_estimators} trees)"

    print(f"Training {clf_name} on {X_train.shape[0]} samples, {X_train.shape[1]} features ...")
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test  accuracy: {test_acc:.4f}")

    # 5-fold CV on the original (pre-SMOTE) full data for a robust estimate.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_n_jobs = n_jobs if classifier == "rf" else 1
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=cv_n_jobs)
    print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print()

    return clf, X_train, X_test, y_train, y_test


def save_model(
    clf,
    le: LabelEncoder,
    feature_columns: list[str],
    output_dir: str | Path = DEFAULT_MODEL_DIR,
) -> Path:
    """Persist model, label encoder, and feature column list to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "rf_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    meta_path = output_dir / "model_meta.json"
    meta = {
        "label_classes": le.classes_.tolist(),
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "classifier": type(clf).__name__,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    le_path = output_dir / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {meta_path}")
    return model_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train SubCell-Predictor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fasta", help="Path to annotated FASTA file (with SL= tags)")
    group.add_argument("--tsv", help="Path to annotated TSV file (from fetch_locations)")
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--classifier", choices=["rf", "gb"], default="rf",
                        help="rf = Random Forest (default), gb = Gradient Boosting")
    parser.add_argument("--smote", action="store_true",
                        help="Apply SMOTE oversampling to balance classes")
    parser.add_argument("--aac-only", action="store_true",
                        help="Use only AAC features (fastest, 20 features)")
    parser.add_argument("--all-features", action="store_true",
                        help="Use all feature sets: AAC + DPC + physicochemical (no k-mers for speed)")
    args = parser.parse_args(argv)

    # Determine feature sets.
    if args.aac_only:
        use_aac, use_dpc, use_kmer, use_phys = True, False, False, False
    elif args.all_features:
        use_aac, use_dpc, use_kmer, use_phys = True, True, False, True
    else:
        use_aac, use_dpc, use_kmer, use_phys = True, False, True, False

    X, y, le = prepare_data(
        fasta_path=args.fasta, tsv_path=args.tsv,
        use_aac=use_aac, use_dpc=use_dpc, use_kmer=use_kmer,
        use_physicochemical=use_phys,
    )
    clf, X_train, X_test, y_train, y_test = train_model(
        X, y,
        classifier=args.classifier,
        use_smote=args.smote,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
    )
    save_model(clf, le, feature_columns=X.columns.tolist(), output_dir=args.output_dir)


if __name__ == "__main__":
    main()
