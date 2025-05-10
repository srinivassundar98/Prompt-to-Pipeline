# transforms.py

import re
import pandas as pd
from pathlib import Path
from langchain.tools import tool
from staging_storage import _get_connection, stage_to_snowflake

@tool
def transform_data(raw_table: str) -> dict:
    """
    1) Load raw_table into DataFrame
    2) Drop nulls
    3) Clean & rename columns and Copy into a new dataframe. Make sure column names are all caps. 'index' should be 'INDEX' etc.
    4) Split out a dimension table on 'Category'
    5) Write both the dim and fact tables back to Snowflake
    Returns both new table names.
    """
    # 1) Load
    conn = _get_connection()
    df = pd.read_sql(f"SELECT * FROM {raw_table}", conn)

    # 2) Drop null rows
    df = df.dropna()

    # 3) Clean & rename columns
    def clean(name: str) -> str:
        name = name.strip(' "')
        name = re.sub(r'[\(\)]', '', name)
        return re.sub(r'[^0-9A-Za-z]+', '_', name).strip('_').lower()
    df.columns = [clean(c) for c in df.columns]

    # ——— 4) Your dimension/fact split ————————————————
    # Example: split out a small dimension on column 'Category'
    if 'category' in df.columns:
        # build dimension
        dim = df[['category']].drop_duplicates().reset_index(drop=True)
        dim_table = stage_to_snowflake(df=dim, table_hint='CATEGORY_DIM')

        # merge back with an integer key
        dim = dim.reset_index().rename(columns={'index': 'category_id'})
        df = df.merge(dim, on='category').drop(columns=['category'])
    else:
        dim_table = None

    # 5) Stage the final fact table
    fact_table = stage_to_snowflake(df=df, table_hint='FACT_MAIN')
    # —————————————————————————————————————————————

    return {
        "dim_table": dim_table,
        "fact_table": fact_table
    }
