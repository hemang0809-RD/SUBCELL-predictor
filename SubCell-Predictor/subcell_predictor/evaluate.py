"""Evaluation: confusion matrix, classification report, ROC curves, PR curves, and feature importance.

Usage:
    python -m subcell_predictor.evaluate --tsv data/annotated_proteins.tsv --save-plots results/
    python -m subcell_predictor.evaluate --tsv data/annotated_proteins.tsv --save-plots results/ --classifier gb --smote --all-features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from subcell_predictor.train import (
    DEFAULT_MODEL_DIR,
    prepare_data,
    train_model,
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
) -> None:
    """Render and optionally save a heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Subcellular Localization — Confusion Matrix")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 30,
    save_path: Path | None = None,
) -> None:
    """Bar chart of the top-N most important features."""
    top_n = min(top_n, len(importances))
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=8)
    ax.set_xlabel("Feature importance (Gini)")
    ax.set_title(f"Top {top_n} Features")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
) -> None:
    """Plot per-class ROC curves with AUC scores."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"ROC curves saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
) -> None:
    """Plot per-class Precision-Recall curves with AUC scores."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{cls_name} (AUC = {pr_auc:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax.legend(loc="lower left")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Precision-Recall curves saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def run_evaluation(
    fasta_path: str | Path | None = None,
    tsv_path: str | Path | None = None,
    save_dir: str | Path | None = None,
    aac_only: bool = False,
    all_features: bool = False,
    classifier: str = "rf",
    use_smote: bool = False,
) -> None:
    """Full evaluation pipeline: train, predict on test set, report metrics."""
    if aac_only:
        use_aac, use_dpc, use_kmer, use_phys = True, False, False, False
    elif all_features:
        use_aac, use_dpc, use_kmer, use_phys = True, True, False, True
    else:
        use_aac, use_dpc, use_kmer, use_phys = True, False, True, False

    X, y, le = prepare_data(
        fasta_path=fasta_path, tsv_path=tsv_path,
        use_aac=use_aac, use_dpc=use_dpc, use_kmer=use_kmer,
        use_physicochemical=use_phys,
    )
    clf, X_train, X_test, y_train, y_test = train_model(
        X, y, classifier=classifier, use_smote=use_smote,
    )

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    class_names = le.classes_.tolist()

    # --- Classification report ---
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- Per-class AUC summary ---
    from sklearn.preprocessing import label_binarize as lb
    y_bin = lb(y_test, classes=list(range(len(class_names))))
    print("Per-class ROC AUC:")
    for i, cls_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        print(f"  {cls_name}: {roc_auc:.4f}")
    print()

    # --- Plots ---
    save_path_cm = None
    save_path_fi = None
    save_path_roc = None
    save_path_pr = None

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path_cm = save_dir / "confusion_matrix.png"
        save_path_fi = save_dir / "feature_importance.png"
        save_path_roc = save_dir / "roc_curves.png"
        save_path_pr = save_dir / "precision_recall_curves.png"

    plot_confusion_matrix(y_test, y_pred, class_names, save_path=save_path_cm)
    plot_feature_importance(
        clf.feature_importances_,
        feature_names=X.columns.tolist(),
        save_path=save_path_fi,
    )
    plot_roc_curves(y_test, y_proba, class_names, save_path=save_path_roc)
    plot_precision_recall_curves(y_test, y_proba, class_names, save_path=save_path_pr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate SubCell-Predictor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fasta", help="Path to annotated FASTA file (with SL= tags)")
    group.add_argument("--tsv", help="Path to annotated TSV file (from fetch_locations)")
    parser.add_argument("--save-plots", default=None, help="Directory to save plots")
    parser.add_argument("--classifier", choices=["rf", "gb"], default="rf",
                        help="rf = Random Forest, gb = Gradient Boosting")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE oversampling")
    parser.add_argument("--aac-only", action="store_true", help="Use only AAC features")
    parser.add_argument("--all-features", action="store_true",
                        help="Use AAC + DPC + physicochemical features")
    args = parser.parse_args(argv)

    run_evaluation(
        fasta_path=args.fasta, tsv_path=args.tsv,
        save_dir=args.save_plots, aac_only=args.aac_only,
        all_features=args.all_features,
        classifier=args.classifier, use_smote=args.smote,
    )


if __name__ == "__main__":
    main()
