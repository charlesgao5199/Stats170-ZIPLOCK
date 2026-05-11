import time
import pandas as pd
import requests
from functools import reduce
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "zcta_archive" / "source"
RAW_DIR = PROJECT_ROOT / "data" / "zcta_archive" / "raw"

CA_ZCTA_PATH = SOURCE_DIR / "ca_zctas_2020.csv"
OUT_PATH = RAW_DIR / "acs_2012_demographics_ca_zcta.csv"
ERROR_PATH = RAW_DIR / "acs_2012_demographics_errors.csv"

API_KEY = "e1721cda47c8fb4390b39599a25a359f7a1df082"
BASE_URL = "https://api.census.gov/data/2012/acs/acs5"

GROUPS = [
    "B01003",  # total population
    "B01001",  # sex by age
    "B02001",  # race
    "B03002",  # hispanic or latino by race
    "B05002",  # place of birth by nativity/citizenship
    "B06009",  # place of birth by educational attainment
    "B15003",  # educational attainment
    "B19013",  # median household income
]

SLEEP_SECONDS = 0
TIMEOUT_SECONDS = 120
RETRIES = 3

# -------------------------------
# Load California ZCTA list
# -------------------------------
ca_zctas = (
    pd.read_csv(CA_ZCTA_PATH, dtype=str)["ZCTA5"]
    .dropna()
    .str.strip()
    .str.zfill(5)
    .drop_duplicates()
)

ca_set = set(ca_zctas.tolist())

session = requests.Session()
group_dfs = []
errors = []

def fetch_group_all_zctas(group_id: str) -> pd.DataFrame:
    """
    Fetch one ACS group for all ZCTAs nationwide, then return as DataFrame.
    """
    params = {
        "get": f"group({group_id})",
        "for": "zip code tabulation area:*",
        "key": API_KEY,
    }

    last_error = None

    for attempt in range(1, RETRIES + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()
            payload = resp.json()

            if not payload or len(payload) < 2:
                raise ValueError(f"Unexpected payload for group {group_id}: {payload}")

            header = payload[0]
            rows = payload[1:]
            df = pd.DataFrame(rows, columns=header)
            # Defensive guard: API can return duplicate labels (e.g., NAME).
            df = df.loc[:, ~df.columns.duplicated()].copy()

            geo_col = "zip code tabulation area"
            if geo_col not in df.columns:
                raise KeyError(f"Missing geography column in group {group_id}: {list(df.columns)}")

            # Normalize ZCTA column
            df[geo_col] = df[geo_col].astype(str).str.zfill(5)

            # Filter to California ZCTAs locally
            df = df[df[geo_col].isin(ca_set)].copy()

            # Rename the geography column to a consistent merge key
            df = df.rename(columns={geo_col: "ZCTA5"})

            return df

        except Exception as e:
            last_error = e
            if attempt < RETRIES:
                time.sleep(SLEEP_SECONDS * attempt)

    raise last_error

# -------------------------------
# Fetch each group once
# -------------------------------
for i, group_id in enumerate(GROUPS, start=1):
    try:
        df_group = fetch_group_all_zctas(group_id)
        group_dfs.append(df_group)
        print(f"[{i}/{len(GROUPS)}] fetched {group_id}: {len(df_group):,} CA rows")
    except Exception as e:
        errors.append({
            "group": group_id,
            "error": str(e),
        })
        print(f"[{i}/{len(GROUPS)}] FAILED {group_id}: {e}")

    time.sleep(SLEEP_SECONDS)

# -------------------------------
# Merge all groups
# -------------------------------
if not group_dfs:
    raise RuntimeError("No group data retrieved.")

# Keep non-key columns unique across all dataframes (e.g., NAME, GEO_ID).
seen_nonkey_cols = set()
for i, df in enumerate(group_dfs):
    overlap = [c for c in df.columns if c != "ZCTA5" and c in seen_nonkey_cols]
    if overlap:
        group_dfs[i] = df.drop(columns=overlap)
    seen_nonkey_cols.update(c for c in group_dfs[i].columns if c != "ZCTA5")

# Merge on ZCTA only.
df_final = reduce(
    lambda left, right: pd.merge(left, right, on=["ZCTA5"], how="outer"),
    group_dfs
)

# Optional: sort by ZCTA
df_final = df_final.sort_values("ZCTA5").reset_index(drop=True)

# Put ZCTA5 first in output columns.
if "ZCTA5" in df_final.columns:
    ordered_cols = ["ZCTA5"] + [c for c in df_final.columns if c != "ZCTA5"]
    df_final = df_final[ordered_cols]

# Save main file
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(OUT_PATH, index=False)

# Save errors if any
if errors:
    pd.DataFrame(errors).to_csv(ERROR_PATH, index=False)

print(f"\nWrote main file: {OUT_PATH}")
print(f"Rows written: {len(df_final):,}")
print(f"Columns written: {len(df_final.columns):,}")
print(f"Failed groups: {len(errors):,}")
if errors:
    print(f"Error file: {ERROR_PATH}")
