"""FASTA parser that extracts protein sequences and subcellular location annotations.

Expects UniProt-style FASTA headers containing subcellular location info, e.g.:
>sp|P12345|PROT_HUMAN ... OS=Homo sapiens ... SL=Nucleus
or headers where subcellular location is embedded as a key-value token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO


# Canonical locations we model — everything else is filtered out.
VALID_LOCATIONS: set[str] = {"Nucleus", "Cytoplasm", "Mitochondrion"}

# Mapping of common UniProt location variants to canonical labels.
_LOCATION_ALIASES: dict[str, str] = {
    "nucleus": "Nucleus",
    "cytoplasm": "Cytoplasm",
    "mitochondrion": "Mitochondrion",
    "mitochondria": "Mitochondrion",
    "cell membrane": "Cell membrane",
    "secreted": "Secreted",
    "endoplasmic reticulum": "Endoplasmic reticulum",
}

# Regex patterns to extract subcellular location from UniProt FASTA headers.
# Pattern 1: explicit "SL=<location>" token (custom-annotated FASTA).
_SL_TAG_RE = re.compile(r"SL=([A-Za-z ]+?)(?:\s+\w+=|\s*$)")
# Pattern 2: "SubcellularLocation:" or "SUBCELLULAR LOCATION" block.
_SUBCELL_BLOCK_RE = re.compile(
    r"(?:Subcellular[_ ]?location|SUBCELLULAR LOCATION)[:\s]+([A-Za-z ]+?)(?:[;.,\s{]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ProteinRecord:
    """A single parsed protein entry."""

    accession: str
    description: str
    sequence: str
    location: str  # canonical label from VALID_LOCATIONS


def _normalise_location(raw: str) -> str | None:
    """Map a raw location string to a canonical label, or ``None`` if unrecognised."""
    key = raw.strip().lower()
    return _LOCATION_ALIASES.get(key)


def _extract_location(header: str) -> str | None:
    """Try to pull a subcellular location from a FASTA description line."""
    # Try the explicit SL= tag first (fastest).
    match = _SL_TAG_RE.search(header)
    if match:
        return _normalise_location(match.group(1))

    # Fall back to the longer 'SubcellularLocation:' block.
    match = _SUBCELL_BLOCK_RE.search(header)
    if match:
        return _normalise_location(match.group(1))

    # Last resort: scan for any known keyword anywhere in the header.
    header_lower = header.lower()
    for alias, canonical in _LOCATION_ALIASES.items():
        if alias in header_lower:
            return canonical

    return None


def parse_fasta(filepath: str | Path) -> list[ProteinRecord]:
    """Parse a UniProt-style FASTA file and return annotated protein records.

    Only proteins whose location maps to one of :data:`VALID_LOCATIONS` are kept.

    Parameters
    ----------
    filepath:
        Path to a ``.fasta`` / ``.fa`` file.

    Returns
    -------
    list[ProteinRecord]
        Parsed records with sequence + canonical subcellular location.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"FASTA file not found: {filepath}")

    records: list[ProteinRecord] = []

    for entry in SeqIO.parse(str(filepath), "fasta"):
        location = _extract_location(entry.description)
        if location is None or location not in VALID_LOCATIONS:
            continue

        # Extract accession (first |-delimited field for sp|XXXXX|… format).
        parts = entry.id.split("|")
        accession = parts[1] if len(parts) >= 2 else entry.id

        records.append(
            ProteinRecord(
                accession=accession,
                description=entry.description,
                sequence=str(entry.seq),
                location=location,
            )
        )

    return records


def records_to_dataframe(records: list[ProteinRecord]):
    """Convert a list of :class:`ProteinRecord` to a :class:`pandas.DataFrame`."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "accession": r.accession,
                "description": r.description,
                "sequence": r.sequence,
                "location": r.location,
            }
            for r in records
        ]
    )


def parse_tsv(filepath: str | Path) -> list[ProteinRecord]:
    """Parse an annotated TSV file (accession, location, sequence).

    This is the preferred input format when working with real UniProt data,
    since subcellular locations are not present in standard FASTA headers.
    Use :mod:`subcell_predictor.fetch_locations` to generate this file.
    """
    import pandas as pd

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TSV file not found: {filepath}")

    df = pd.read_csv(filepath, sep="\t")
    required = {"accession", "location", "sequence"}
    if not required.issubset(df.columns):
        raise ValueError(f"TSV must contain columns: {required}. Found: {set(df.columns)}")

    records: list[ProteinRecord] = []
    for _, row in df.iterrows():
        loc = row["location"]
        if loc not in VALID_LOCATIONS:
            continue
        records.append(
            ProteinRecord(
                accession=row["accession"],
                description="",
                sequence=row["sequence"],
                location=loc,
            )
        )
    return records
