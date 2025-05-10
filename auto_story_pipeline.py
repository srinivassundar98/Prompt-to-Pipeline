
# auto_story_pipeline.py

import os
import re
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from ingestion_engine import ingest
from staging_storage import stage_to_snowflake
from analysis_suggester import suggest_analyses
from analysis_runner import run_analysis
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

def _infer_schema(df: pd.DataFrame) -> list[dict]:
    """Infer simple JSON schema from a pandas DataFrame."""
    type_map = {
        "int64": "INTEGER",
        "float64": "FLOAT",
        "object": "STRING",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
    }
    schema = []
    for name, dtype in df.dtypes.items():
        schema.append({"name": name, "type": type_map.get(str(dtype), "STRING")})
    return schema

def auto_pipeline(url: str, table_hint: str = None):
    # ─── 1) Ingest & stage raw data ────────────────────────────────────────────
    meta = ingest(url)
    raw_summary = stage_to_snowflake.invoke({
        "df": meta["df"],
        "table_hint": table_hint
    })
    print(raw_summary)

    # ─── 2) In‐memory TRANSFORM ────────────────────────────────────────────────
    df = meta["df"]

    # 2.1) Drop nulls
    before = len(df)
    df = df.dropna()
    print(f"Dropped rows with nulls: {before - len(df)} removed → {len(df)} rows remain")

    # 2.2) Clean & rename columns (strip spaces/quotes/parens → UPPER_SNAKE)
    def clean_col(c: str) -> str:
        c = c.strip(' "')
        c = re.sub(r'[\(\)]', '', c)
        c = re.sub(r'[^0-9A-Za-z]+', '_', c).strip('_').upper()
        return c

    old_cols = df.columns.tolist()
    df.columns = [clean_col(c) for c in old_cols]
    print(f"Renamed columns:\n  {old_cols!r}\n→ {df.columns.tolist()!r}")

    # 2.3) Normalize categorical columns into dimension tables
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        dim_df = df[[col]].drop_duplicates().reset_index(drop=True)
        hint = f"{table_hint}_{col}_dim" if table_hint else f"{col}_dim"
        dim_summary = stage_to_snowflake.invoke({"df": dim_df, "table_hint": hint})
        m = re.search(r" to (\S+)\s*\(", dim_summary)
        dim_tbl = m.group(1) if m else None
        print(f"Staged dimension '{col}' → {dim_tbl}")

        key_df = dim_df.reset_index().rename(columns={"index": f"{col}_ID"})
        df = df.merge(key_df, on=col).drop(columns=[col])

    # 2.4) Stage fact table
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fact_hint = f"{table_hint}_fact_{ts}" if table_hint else f"fact_{ts}"
    fact_summary = stage_to_snowflake.invoke({
        "df": df,
        "table_hint": fact_hint
    })
    print(f"Staged fact table → {fact_summary}")
    m = re.search(r" to (\S+)\s*\(", fact_summary)
    fact_table = m.group(1) if m else None
    Path(".last_table").write_text(fact_table)

    # ─── 3) Suggest analyses on cleaned schema ────────────────────────────────
    schema = _infer_schema(df)
    schema_str = ", ".join(f"{c['name']}({c['type']})" for c in schema)
    llm = ChatOllama(model="llama3:8b", temperature=0.3)
    prompt_suggest = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior data scientist.\n"
         f"The cleaned schema is: {schema_str}.\n"
         "List 12 insightful analyses the user could run next. "
         "Return each idea as '- idea' on its own line."),
        ("human", "")
    ])
    raw_ideas = llm.invoke(prompt_suggest.format()).content.strip()
    ideas = [line[2:].strip() for line in raw_ideas.splitlines() if line.startswith("-")]
    print("\nGenerated analysis ideas:")
    for i, idea in enumerate(ideas, 1):
        print(f"{i}. {idea}")

    # ─── 4) Run each analysis and collect chart HTMLs ─────────────────────────
    chart_info = []
    for idea in ideas:
        print(f"\nRunning: {idea}")
        html_path = run_analysis.invoke({"request": idea})
        chart_info.append((idea, html_path))
        print(f" → HTML: {html_path}")

        # ─── 5) Describe each existing JSON file ─────────────────────────────────


    # ─── 5) Describe each existing JSON file ─────────────────────────────────
    json_dir = Path("charts2/json")
    desc_dir = Path("descriptions")
    desc_dir.mkdir(exist_ok=True)
    chart_descriptions = []

    def sanitize_plotly_json(raw_json: str) -> str:
        obj = json.loads(raw_json)
        # strip out heavy 'bdata' and 'dtype' fields from each trace
        for trace in obj.get("data", []):
            for axis in ("x", "y"):
                val = trace.get(axis)
                if isinstance(val, dict):
                    val.pop("bdata", None)
                    val.pop("dtype", None)
        # remove the full layout template
        obj.get("layout", {}).pop("template", None)
        return json.dumps(obj)

    for jsn_file in sorted(json_dir.glob("*.json")):
        # derive a human‐friendly idea from the filename
        safe = jsn_file.stem.split("_", 2)[-1]
        idea = safe.replace("_", " ")
        
        # read & sanitize the raw JSON
        raw_json = jsn_file.read_text()
        sanitized_json = sanitize_plotly_json(raw_json)
        
        # build the LLM prompt
        desc_prompt = (
            f"You are a data storyteller.\n"
            f"Here is the sanitized Plotly JSON for the chart titled '{idea}':\n"
            f"```json\n{sanitized_json}\n```\n"
            "Write a one-sentence description of what this chart reveals about the data."
        )
        
        # save the prompt
        (desc_dir / f"{safe}_prompt.txt").write_text(desc_prompt)
        
        # invoke the LLM
        desc = llm.invoke(desc_prompt).content.strip()
        
        # save the description
        (desc_dir / f"{safe}_description.txt").write_text(desc)
        
        # collect for later story weaving
        chart_descriptions.append(f"- **{idea}**: {desc}")


    # ─── 6) Weave narrative from descriptions ────────────────────────────────
    story_prompt = (
        "You are a data storyteller. You have the following chart summaries:\n"
        + "\n".join(chart_descriptions)
        + "\n\nWrite a concise narrative that ties these insights together."
    )
    (Path("descriptions") / "narrative_prompt.txt").write_text(story_prompt)
    story = llm.invoke(story_prompt).content.strip()
    (Path("descriptions") / "narrative.txt").write_text(story)

    # ─── 7) Output ───────────────────────────────────────────────────────────
    print("\n=== Narrative ===")
    print(story)
    print("\nCharts saved at:")
    for _, path in chart_info:
        print(f" - {path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--url",  type=str,  required=True)
    p.add_argument("--hint", type=str,  dest="table_hint")
    args = p.parse_args()
    auto_pipeline(args.url, args.table_hint)
