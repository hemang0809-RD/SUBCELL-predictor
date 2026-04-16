"""Fetch subcellular location annotations from the UniProt REST API.

UniProt FASTA headers do not include subcellular location, so we fetch it
separately as a TSV and merge with the parsed FASTA sequences.

Usage:
    python -m subcell_predictor.fetch_locations --fasta data/uniprot_human.fasta --output data/annotated_proteins.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import requests
from Bio import SeqIO

VALID_LOCATIONS: set[str] = {"Nucleus", "Cytoplasm", "Mitochondrion"}

_LOCATION_ALIASES: dict[str, str] = {
    "nucleus": "Nucleus",
    "cytoplasm": "Cytoplasm",
    "cytosol": "Cytoplasm",
    "mitochondrion": "Mitochondrion",
    "mitochondria": "Mitochondrion",
    "mitochondrial matrix": "Mitochondrion",
    "mitochondrion inner membrane": "Mitochondrion",
    "mitochondrion outer membrane": "Mitochondrion",
}

# UniProt REST API base URL for streaming results.
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/stream"

# How many accessions to send per API request (UniProt has URL length limits).
BATCH_SIZE = 100


def _classify_location(raw_location: str) -> str | None:
    """Parse UniProt's 'Subcellular location [CC]' field and return a canonical label.

    UniProt location fields look like:
        "SUBCELLULAR LOCATION: Nucleus. Cytoplasm."
        "SUBCELLULAR LOCATION: Mitochondrion inner membrane; Multi-pass..."
    We take the *first* matching canonical location.
    """
    raw_lower = raw_location.lower()
    for alias, canonical in _LOCATION_ALIASES.items():
        if alias in raw_lower:
            if canonical in VALID_LOCATIONS:
                return canonical
    return None


def _extract_accessions_from_fasta(fasta_path: Path) -> list[str]:
    """Read all accessions from a UniProt FASTA file."""
    accessions = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        parts = record.id.split("|")
        acc = parts[1] if len(parts) >= 2 else record.id
        accessions.append(acc)
    return accessions


def fetch_locations_batch(accessions: list[str]) -> dict[str, str]:
    """Fetch subcellular locations for a batch of accessions from UniProt API.

    Returns a dict mapping accession -> canonical location label.
    """
    query = " OR ".join(f"accession:{acc}" for acc in accessions)
    params = {
        "query": query,
        "fields": "accession,cc_subcellular_location",
        "format": "tsv",
    }

    resp = requests.get(UNIPROT_API, params=params, timeout=120)
    resp.raise_for_status()

    results: dict[str, str] = {}
    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        return results

    reader = csv.DictReader(lines, delimiter="\t")
    for row in reader:
        acc = row.get("Entry", "").strip()
        loc_field = row.get("Subcellular location [CC]", "").strip()
        if not acc or not loc_field:
            continue
        canonical = _classify_location(loc_field)
        if canonical:
            results[acc] = canonical

    return results


def fetch_all_locations(accessions: list[str]) -> dict[str, str]:
    """Fetch subcellular locations for all accessions, batched."""
    all_results: dict[str, str] = {}
    total = len(accessions)

    for i in range(0, total, BATCH_SIZE):
        batch = accessions[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Fetching batch {batch_num}/{total_batches} ({len(batch)} accessions)...")

        results = fetch_locations_batch(batch)
        all_results.update(results)

        # Rate-limit: be polite to UniProt's servers.
        if i + BATCH_SIZE < total:
            time.sleep(1.0)

    return all_results


def build_annotated_tsv(
    fasta_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Parse a FASTA file, fetch locations from UniProt, and write an annotated TSV.

    The TSV has columns: accession, location, sequence
    Only proteins with a valid location (Nucleus/Cytoplasm/Mitochondrion) are kept.
    """
    fasta_path = Path(fasta_path)
    output_path = Path(output_path)

    print(f"Reading accessions from {fasta_path}...")
    accessions = _extract_accessions_from_fasta(fasta_path)
    print(f"  Found {len(accessions)} proteins in FASTA.")

    print("Fetching subcellular locations from UniProt API...")
    location_map = fetch_all_locations(accessions)
    print(f"  Got locations for {len(location_map)} proteins in target classes.")

    # Now merge: re-read FASTA for sequences, write only annotated ones.
    print(f"Writing annotated data to {output_path}...")
    count = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["accession", "location", "sequence"])

        for record in SeqIO.parse(str(fasta_path), "fasta"):
            parts = record.id.split("|")
            acc = parts[1] if len(parts) >= 2 else record.id

            if acc in location_map:
                writer.writerow([acc, location_map[acc], str(record.seq)])
                count += 1

    print(f"  Wrote {count} annotated proteins.")
    print()

    # Summary.
    from collections import Counter
    counts = Counter(location_map.values())
    print("Class distribution:")
    for loc, n in sorted(counts.items()):
        print(f"  {loc}: {n}")

    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch subcellular locations from UniProt and build annotated TSV"
    )
    parser.add_argument("--fasta", required=True, help="Path to UniProt FASTA file")
    parser.add_argument(
        "--output",
        default="data/annotated_proteins.tsv",
        help="Output TSV path (default: data/annotated_proteins.tsv)",
    )
    args = parser.parse_args(argv)

    build_annotated_tsv(args.fasta, args.output)


if __name__ == "__main__":
    main()
