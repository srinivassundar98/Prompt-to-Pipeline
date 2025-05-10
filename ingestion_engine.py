"""ingestion_engine.py

Light‑weight ingestion helper for Prompt‑to‑Pipeline demo.

Exposed public function
-----------------------
    ingest(url: str) -> dict
        Downloads a publicly reachable file (<= ~3 GB), loads it into a
        pandas DataFrame, infers a simple JSON schema, and returns metadata.

Returned dict schema
--------------------
{
    "local_path": str,     # absolute path to temp file
    "df": pandas.DataFrame, # the loaded data (small previews only)
    "schema": list[dict],  # [{"name": "col", "type": "STRING"}, ...]
    "rows": int            # number of rows loaded
}

Notes
-----
* Supports CSV and Parquet URLs out‑of‑the‑box.  Additional formats can be
  added by extending `_load_dataframe`.
* Does *not* keep large DataFrames in memory during production use; caller
  should persist to warehouse immediately and drop the reference.
"""

from __future__ import annotations

import json
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download_file(url: str) -> Path:
    """Stream‑download *url* into a secure temporary file and return the path."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    suffix = Path(url).suffix or mimetypes.guess_extension(response.headers.get("Content‑Type", "")) or ".dat"
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(tmp_fd, "wb") as fp:
        for chunk in response.iter_content(chunk_size=1 << 20):  # 1 MiB
            fp.write(chunk)
    return Path(tmp_name)


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Load *path* into a DataFrame based on its extension."""
    ext = path.suffix.lower()
    if ext in {".csv", ".txt"}:
        return pd.read_csv(path)
    if ext in {".parquet", " .pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def _infer_schema(df: pd.DataFrame) -> List[Dict[str, str]]:
    type_map = {
        "int64": "INTEGER",
        "float64": "FLOAT",
        "object": "STRING",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
    }
    schema = []
    for name, dtype in df.dtypes.items():
        schema.append({
            "name": name,
            "type": type_map.get(str(dtype), "STRING"),
        })
    return schema


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest(url: str) -> Dict:
    """Download a file from *url*, load, infer schema, return metadata."""
    local_path = _download_file(url)
    df = _load_dataframe(local_path)
    schema = _infer_schema(df)
    return {
        "local_path": str(local_path),
        "df": df,  # caller can sample/preview; drop after staging
        "schema": schema,
        "rows": len(df),
    }

# ---------------------------------------------------------------------------
# CLI test (run `python ingestion_engine.py <url>`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pprint
    if len(sys.argv) < 2:
        print("Usage: python ingestion_engine.py <public-data-url>")
        sys.exit(1)
    url_arg = sys.argv[1]
    meta = ingest(url_arg)
    print("\nLoaded", meta["rows"], "rows from", url_arg)
    print("Schema:")
    pprint.pp(meta["schema"])
