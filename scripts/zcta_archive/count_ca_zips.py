#!/usr/bin/env python3
"""Count California rows and unique California ZIP/ZCTA codes."""

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "zcta_archive" / "source" / "2020_ZCTA_relationship.txt"


def count_ca_rows(
    path: str,
    state_col: str,
    county_col: str,
    zip_col: str,
    ca_fips: str,
) -> tuple[int, int]:
    row_count = 0
    unique_zips = set()

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")

        if reader.fieldnames is None:
            raise ValueError("Input file has no header row.")

        fieldnames = set(reader.fieldnames)
        has_state = state_col in fieldnames
        has_county = county_col in fieldnames
        has_zip = zip_col in fieldnames
        ca_fips = ca_fips.zfill(2)

        if not has_state and not has_county:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Neither '{state_col}' nor '{county_col}' found. "
                f"Available columns: {available}"
            )
        if not has_zip:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"ZIP/ZCTA column '{zip_col}' not found. Available columns: {available}"
            )

        for row in reader:
            is_ca = False
            if has_state:
                state_val = (row.get(state_col) or "").strip().zfill(2)
                if state_val == ca_fips:
                    is_ca = True
            else:
                # County GEOID is SSCCC; California counties start with "06".
                county_val = (row.get(county_col) or "").strip().zfill(5)
                if county_val.startswith(ca_fips):
                    is_ca = True

            if is_ca:
                row_count += 1
                raw_zip = (row.get(zip_col) or "").strip()
                if raw_zip:
                    unique_zips.add(raw_zip.zfill(5))

    return row_count, len(unique_zips)


def main() -> int:
    parser = argparse.ArgumentParser(description="Count California rows in a file.")
    parser.add_argument(
        "file",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=f"Input file path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--state-col",
        default="STATEFP",
        help="State FIPS column name (default: STATEFP)",
    )
    parser.add_argument(
        "--county-col",
        default="GEOID_COUNTY_20",
        help="County GEOID column name (default: GEOID_COUNTY_20)",
    )
    parser.add_argument(
        "--zip-col",
        default="GEOID_ZCTA5_20",
        help="ZIP/ZCTA code column name (default: GEOID_ZCTA5_20)",
    )
    parser.add_argument(
        "--ca-fips",
        default="06",
        help='California state FIPS code (default: "06")',
    )
    args = parser.parse_args()

    try:
        row_count, unique_zip_count = count_ca_rows(
            args.file,
            args.state_col,
            args.county_col,
            args.zip_col,
            args.ca_fips,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"CA rows: {row_count}")
    print(f"Unique CA ZIP/ZCTA codes: {unique_zip_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
