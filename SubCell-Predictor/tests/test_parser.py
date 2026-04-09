"""Unit tests for the FASTA parser."""

import tempfile
from pathlib import Path

import pytest

from subcell_predictor.parser import (
    ProteinRecord,
    _extract_location,
    parse_fasta,
    records_to_dataframe,
)

SAMPLE_FASTA = """\
>sp|P12345|PROT1_HUMAN Some protein OS=Homo sapiens SL=Nucleus OX=9606
MSTIKLGLLVLFVAQFALLSQAGSTAEGKLIVEDD
>sp|P67890|PROT2_HUMAN Another protein OS=Homo sapiens SL=Cytoplasm OX=9606
ACDEFGHIKLMNPQRSTVWY
>sp|P11111|PROT3_HUMAN Mito protein OS=Homo sapiens SL=Mitochondrion OX=9606
MSTIKLGLLVLFVAQFALLSQAGSWYWYWY
>sp|P99999|PROT4_HUMAN Secreted protein OS=Homo sapiens SL=Secreted OX=9606
ACDEFGHIKLMNPQ
"""


@pytest.fixture()
def fasta_path(tmp_path: Path) -> Path:
    fpath = tmp_path / "test.fasta"
    fpath.write_text(SAMPLE_FASTA)
    return fpath


def test_extract_location_sl_tag():
    header = "sp|P12345|X OS=Homo sapiens SL=Nucleus OX=9606"
    assert _extract_location(header) == "Nucleus"


def test_extract_location_returns_none_for_unknown():
    assert _extract_location("sp|X|Y some random header") is None


def test_parse_fasta_filters_valid_locations(fasta_path: Path):
    records = parse_fasta(fasta_path)
    # Secreted is not in VALID_LOCATIONS, so only 3 records.
    assert len(records) == 3
    locations = {r.location for r in records}
    assert locations == {"Nucleus", "Cytoplasm", "Mitochondrion"}


def test_parse_fasta_extracts_accession(fasta_path: Path):
    records = parse_fasta(fasta_path)
    accessions = {r.accession for r in records}
    assert "P12345" in accessions


def test_records_to_dataframe(fasta_path: Path):
    records = parse_fasta(fasta_path)
    df = records_to_dataframe(records)
    assert list(df.columns) == ["accession", "description", "sequence", "location"]
    assert len(df) == 3
