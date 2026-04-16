"""End-to-end integration test: parse → featurize → train → evaluate → predict."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from subcell_predictor.evaluate import run_evaluation
from subcell_predictor.train import prepare_data, save_model, train_model


def _generate_synthetic_fasta(path: Path, n_per_class: int = 30) -> None:
    """Write a synthetic FASTA file with known locations and distinct sequence biases."""
    import random

    random.seed(42)

    classes = {
        "Nucleus": "KKRRKKRRHHPP",    # basic / NLS-like bias
        "Cytoplasm": "AAGGLLIIVVSS",  # small/hydrophobic bias
        "Mitochondrion": "LLFFWWMMYY", # hydrophobic / MTS-like bias
    }

    with open(path, "w") as f:
        idx = 0
        for loc, alphabet in classes.items():
            for i in range(n_per_class):
                idx += 1
                seq_len = random.randint(80, 200)
                seq = "".join(random.choices(alphabet, k=seq_len))
                f.write(f">sp|P{idx:05d}|SYN{idx}_HUMAN Synthetic OS=Homo sapiens SL={loc} OX=9606\n")
                f.write(seq + "\n")


@pytest.fixture()
def synthetic_fasta(tmp_path: Path) -> Path:
    fpath = tmp_path / "synthetic.fasta"
    _generate_synthetic_fasta(fpath, n_per_class=40)
    return fpath


def test_full_pipeline(synthetic_fasta: Path, tmp_path: Path):
    """Train on synthetic data and verify that the model can predict."""
    # Use AAC-only for speed (skip 8000-feature k-mer matrix).
    X, y, le = prepare_data(synthetic_fasta, use_aac=True, use_kmer=False)
    assert X.shape[0] == 120  # 40 * 3 classes
    assert X.shape[1] == 20

    clf, X_train, X_test, y_train, y_test = train_model(X, y, test_size=0.25)

    # With strongly biased synthetic data, accuracy should be high.
    test_acc = clf.score(X_test, y_test)
    assert test_acc > 0.7, f"Accuracy too low on synthetic data: {test_acc}"

    # Save and reload.
    model_dir = tmp_path / "models"
    save_model(clf, le, feature_columns=X.columns.tolist(), output_dir=model_dir)
    assert (model_dir / "rf_model.pkl").exists()
    assert (model_dir / "model_meta.json").exists()
    assert (model_dir / "label_encoder.pkl").exists()

    meta = json.loads((model_dir / "model_meta.json").read_text())
    assert set(meta["label_classes"]) == {"Nucleus", "Cytoplasm", "Mitochondrion"}


def test_predict_script(synthetic_fasta: Path, tmp_path: Path):
    """Verify predict.py can load a model and return predictions."""
    from predict import predict_sequences

    X, y, le = prepare_data(synthetic_fasta, use_aac=True, use_kmer=False)
    clf, *_ = train_model(X, y)
    model_dir = tmp_path / "models"
    save_model(clf, le, feature_columns=X.columns.tolist(), output_dir=model_dir)

    results = predict_sequences(
        sequences=["KKRRKKRRHHPPKKRR", "AAGGLLIIVVSSAAGG"],
        names=["nuclear_query", "cytoplasm_query"],
        model_dir=model_dir,
    )
    assert len(results) == 2
    assert "predicted_location" in results.columns
    assert "confidence" in results.columns
    # Nuclear-biased sequence should predict Nucleus.
    assert results.iloc[0]["predicted_location"] == "Nucleus"
