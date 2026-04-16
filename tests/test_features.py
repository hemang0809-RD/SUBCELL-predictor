"""Unit tests for the featurization module."""

import math

import pytest

from subcell_predictor.features import (
    AMINO_ACIDS,
    amino_acid_composition,
    build_feature_matrix,
    compute_aac_matrix,
    compute_dpc_matrix,
    compute_kmer_matrix,
    compute_physicochemical_matrix,
    dipeptide_composition,
    kmer_frequencies,
    physicochemical_features,
)


def test_aac_simple_sequence():
    seq = "A" * 10 + "G" * 10
    comp = amino_acid_composition(seq)
    assert math.isclose(comp["A"], 50.0)
    assert math.isclose(comp["G"], 50.0)
    assert math.isclose(comp["C"], 0.0)


def test_aac_ignores_nonstandard():
    seq = "AAAX"
    comp = amino_acid_composition(seq)
    assert math.isclose(comp["A"], 100.0)


def test_aac_matrix_shape():
    seqs = ["ACDEF", "GHIKL", "MNPQR"]
    df = compute_aac_matrix(seqs)
    assert df.shape == (3, 20)
    assert all(col.startswith("AAC_") for col in df.columns)


def test_dipeptide_composition_basic():
    seq = "AACC"  # dipeptides: AA, AC, CC → 3 total
    freq = dipeptide_composition(seq)
    assert math.isclose(freq["AA"], 1 / 3)
    assert math.isclose(freq["AC"], 1 / 3)
    assert math.isclose(freq["CC"], 1 / 3)


def test_dpc_matrix_shape():
    seqs = ["ACDEFGHIKLMNPQRSTVWY"]
    df = compute_dpc_matrix(seqs)
    assert df.shape == (1, 400)
    assert all(col.startswith("DPC_") for col in df.columns)


def test_kmer_frequencies_basic():
    seq = "AAACCC"
    freq = kmer_frequencies(seq, k=3)
    assert math.isclose(freq["AAA"], 0.25)
    assert math.isclose(freq["CCC"], 0.25)
    assert "AAG" not in freq


def test_kmer_matrix_shape():
    seqs = ["ACDEFGHIKLMNPQRSTVWY"]
    df = compute_kmer_matrix(seqs, k=3)
    assert df.shape == (1, 20**3)
    assert all(col.startswith("KM_") for col in df.columns)


def test_physicochemical_features_basic():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    feats = physicochemical_features(seq)
    assert feats["PC_length"] == 20
    assert feats["PC_mw"] > 0
    assert "PC_avg_hydrophobicity" in feats
    assert "PC_frac_aromatic" in feats


def test_physicochemical_matrix_shape():
    seqs = ["ACDEF", "GHIKL"]
    df = compute_physicochemical_matrix(seqs)
    assert df.shape[0] == 2
    assert df.shape[1] == 12
    assert all(col.startswith("PC_") for col in df.columns)


def test_build_feature_matrix_combined():
    seqs = ["ACDEFGHIKLMNPQRSTVWY", "AAAACCCCDDDDEEEE"]
    df = build_feature_matrix(seqs, use_aac=True, use_kmer=True, k=3)
    assert df.shape[0] == 2
    assert df.shape[1] == 20 + 20**3


def test_build_feature_matrix_aac_only():
    seqs = ["ACDEF"]
    df = build_feature_matrix(seqs, use_aac=True, use_kmer=False)
    assert df.shape == (1, 20)


def test_build_feature_matrix_all_features():
    seqs = ["ACDEFGHIKLMNPQRSTVWY"]
    df = build_feature_matrix(
        seqs, use_aac=True, use_dpc=True, use_kmer=False, use_physicochemical=True
    )
    assert df.shape == (1, 20 + 400 + 12)
