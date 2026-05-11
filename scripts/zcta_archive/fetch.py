import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "zcta_archive" / "source"

INPUT_PATH = SOURCE_DIR / "2020_ZCTA_relationship.txt"
OUTPUT_PATH = SOURCE_DIR / "ca_zctas_2020.csv"

# Read the relationship file (pipe-delimited)
df = pd.read_csv(INPUT_PATH, sep="|", dtype=str, low_memory=False)

# --- Robustly find the needed columns (handles slight name variants) ---
def pick_col(possible):
    for c in possible:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns were found: {possible}\nAvailable columns: {list(df.columns)}")

col_zcta = pick_col(["GEOID_ZCTA5_20", "ZCTA5", "GEOID_ZCTA5"])
col_county = pick_col(["GEOID_COUNTY_20", "GEOID_COUNTY", "COUNTY", "COUNTYFP"])
col_state = "STATEFP" if "STATEFP" in df.columns else None

# --- Filter to California ---
if col_state is not None:
    # STATEFP is 2-digit state FIPS; California = "06"
    ca = df[df[col_state].str.zfill(2).eq("06")]
else:
    # County GEOID is 5-digit: SSCCC; California counties start with "06"
    # If the file has only COUNTYFP (3-digit), you can't infer CA without STATEFP.
    # Here we assume GEOID_COUNTY_* exists (5-digit).
    ca = df[df[col_county].str.zfill(5).str.startswith("06")]

# --- Extract unique 5-digit ZCTAs ---
ca_zctas = (
    ca[col_zcta]
    .dropna()
    .str.strip()
    .str.zfill(5)
    .drop_duplicates()
    .sort_values()
)

# Save
out = pd.DataFrame({"ZCTA5": ca_zctas})
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

print(f"Rows in file: {len(df):,}")
print(f"CA relationship rows: {len(ca):,}")
print(f"Unique CA ZCTAs: {len(out):,}")
print(f"Wrote: {OUTPUT_PATH}")
