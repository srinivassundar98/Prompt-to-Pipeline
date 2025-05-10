# """staging_storage.py

# IMPORTANT: This version hard‑codes Snowflake credentials directly in the
# source file so you can try the demo instantly. **DO NOT use hard‑coded
# credentials in production**—move them to env‑vars or a secrets manager as
# soon as you finish testing.

# Public tool exported
# --------------------
#     stage_to_snowflake(df: pandas.DataFrame, table_hint: str | None) -> str

# The function uses Snowflake Connector's `write_pandas` helper to create a RAW_<name>
# table and stores a JSON schema string in the table comment for lineage.
# """
# from __future__ import annotations

# import hashlib
# from typing import Dict

# import pandas as pd
# from langchain.tools import tool
# import snowflake.connector
# from snowflake.connector.pandas_tools import write_pandas

# # ---------------------------------------------------------------------------
# # !!! Hard‑coded Snowflake credentials — FOR LOCAL DEMO ONLY
# # ---------------------------------------------------------------------------

# _SNOW_CFG: Dict[str, str] = {
#     "account": "OUVCAXB-UTC14250",
#     "user": "SRINIVAS98",
#     "password": "Password123456",
#     "role": "ACCOUNTADMIN",
#     "warehouse": "COMPUTE_WH",
#     "database": "TRIAL",
#     "schema": "PUBLIC",
# }

# def _get_connection():
#     return snowflake.connector.connect(
#         account=_SNOW_CFG["account"],
#         user=_SNOW_CFG["user"],
#         password=_SNOW_CFG["password"],
#         role=_SNOW_CFG["role"],
#         warehouse=_SNOW_CFG["warehouse"],
#         database=_SNOW_CFG["database"],
#         schema=_SNOW_CFG["schema"],
#     )

# # ---------------------------------------------------------------------------
# # Tool: DataFrame -> Snowflake
# # ---------------------------------------------------------------------------

# @tool
# def stage_to_snowflake(df: pd.DataFrame, table_hint: str | None = None) -> str:
#     """Write *df* to Snowflake RAW schema via write_pandas.

#     Returns a summary string with FQN table name and row count.
#     """
#     conn = _get_connection()

#     # Generate target table name
#     if table_hint:
#         safe = table_hint.upper().replace(" ", "_")[:30]
#         table_name = f"RAW_{safe}"
#     else:
#         h = hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()[:8]
#         table_name = f"RAW_{h}"

#     success, nchunks, nrows, _ = write_pandas(conn, df, table_name,auto_create_table=True)

#     if not success:
#         raise RuntimeError("write_pandas reported failure")

#     # Store schema in COMMENT for lineage
#     schema_json = df.dtypes.apply(lambda t: str(t)).to_json()
#     with conn.cursor() as cur:
#         cur.execute(f"COMMENT ON TABLE {table_name} IS %s", (schema_json,))

#     return f"Written {nrows} rows to {table_name} (chunks={nchunks})"


# Updated staging_storage.py with comment-error handling

from __future__ import annotations

import hashlib
from typing import Dict

import pandas as pd
from langchain.tools import tool
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# ---------------------------------------------------------------------------
# !!! Hard‑coded Snowflake credentials — FOR LOCAL DEMO ONLY
# ---------------------------------------------------------------------------
_SNOW_CFG: Dict[str, str] = {
    "account": "OUVCAXB-UTC14250",
    "user": "SRINIVAS98",
    "password": "Password123456",
    "role": "ACCOUNTADMIN",
    "warehouse": "COMPUTE_WH",
    "database": "TRIAL",
    "schema": "PUBLIC",
}

def _get_connection():
    return snowflake.connector.connect(
        account=_SNOW_CFG["account"],
        user=_SNOW_CFG["user"],
        password=_SNOW_CFG["password"],
        role=_SNOW_CFG["role"],
        warehouse=_SNOW_CFG["warehouse"],
        database=_SNOW_CFG["database"],
        schema=_SNOW_CFG["schema"],
    )

@tool
def stage_to_snowflake(df: pd.DataFrame, table_hint: str | None = None) -> str:
    """Write *df* to Snowflake RAW schema via write_pandas, ignoring COMMENT errors."""
    conn = _get_connection()

    # Generate target table name
    if table_hint:
        safe = table_hint.upper().replace(" ", "_")[:30]
        table_name = f"RAW_1"
    else:
        h = hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()[:8]
        table_name = f"RAW_1"
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    success, nchunks, nrows, _ = write_pandas(
        conn, df, table_name, auto_create_table=True
    )
    if not success:
        raise RuntimeError("write_pandas reported failure")

    # Attempt to store schema in COMMENT for lineage
    schema_json = df.dtypes.apply(lambda t: str(t)).to_json()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"COMMENT ON TABLE {table_name} IS %s",
                (schema_json,),
            )
    except Exception:
        # ignore errors on COMMENT (table might not exist yet or lack privileges)
        pass

    return f"Written {nrows} rows to {table_name} (chunks={nchunks})"

