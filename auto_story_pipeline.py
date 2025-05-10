# # auto_story_pipeline.py

# from ingestion_engine import ingest
# from staging_storage import stage_to_snowflake
# from analysis_suggester import suggest_analyses
# from analysis_runner import run_analysis
# from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate
# from pathlib import Path

# def auto_pipeline(url: str, table_hint: str = None):
#     # 1) Ingest and stage
#     meta = ingest(url)
#     summary = stage_to_snowflake.invoke({
#         "df": meta["df"],
#         "table_hint": table_hint
#     })
#     print(summary)

#     # 2) Build schema string
#     schema = ", ".join(f"{c['name']}({c['type']})" for c in meta["schema"])

#     # 3) Generate 12 visualizations suggestions
#     llm = ChatOllama(model="llama3:8b", temperature=0.3)
#     prompt_suggest = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a senior data scientist.  "
#          "Given the schema: " + schema + ", list 12 insightful analyses the user could run next.  "
#          "Return each idea as '- idea' on its own line."),
#         ("human", "")
#     ])
#     raw_ideas = llm.invoke(prompt_suggest.format()).content.strip()
#     ideas = [line[2:].strip() for line in raw_ideas.splitlines() if line.startswith("-")]
#     print("Generated ideas:")
#     for i, idea in enumerate(ideas, 1):
#         print(f"{i}. {idea}")

#     # 4) Render each idea into a chart
#     chart_info = []
#     for idea in ideas:
#         print(f"Running analysis for: {idea}")
#         out = run_analysis.invoke({"request": idea})
#         chart_info.append((idea, out))
#         print(f" → {out}")

#     # 5) Build a narrative story
#     prompt_story = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a data storyteller.  You have generated the following analyses:\n"
#          + "\n".join(f"- {idea}: {path}" for idea, path in chart_info)
#          + "\n\nWrite a concise narrative that connects these insights into a coherent story for the user."),
#         ("human", "")
#     ])
#     story = llm.invoke(prompt_story.format()).content.strip()

#     return chart_info, story

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--url",  type=str, required=True)
#     parser.add_argument("--hint", type=str, dest="table_hint")
#     args = parser.parse_args()

#     charts, narrative = auto_pipeline(args.url, args.table_hint)
#     # Save narrative to file
#     Path("story.txt").write_text(narrative)
#     print("\n=== Narrative Story ===")
#     print(narrative)
#     print("\nCharts saved at:")
#     for _, path in charts:
#         print(f" - {path}")

# # To run:
# # python auto_story_pipeline.py --url "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"

# Updated auto_story_pipeline.py with Transform step
# auto_story_pipeline.py
#######THIS WORKS######
# import re
# from pathlib import Path
# from datetime import datetime

# import pandas as pd
# from ingestion_engine import ingest
# from staging_storage import stage_to_snowflake
# from analysis_suggester import suggest_analyses
# from analysis_runner import run_analysis
# from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate

# def _infer_schema(df: pd.DataFrame) -> list[dict]:
#     """Infer simple JSON schema from a pandas DataFrame."""
#     type_map = {
#         "int64": "INTEGER",
#         "float64": "FLOAT",
#         "object": "STRING",
#         "bool": "BOOLEAN",
#         "datetime64[ns]": "TIMESTAMP",
#     }
#     schema = []
#     for name, dtype in df.dtypes.items():
#         schema.append({"name": name, "type": type_map.get(str(dtype), "STRING")})
#     return schema

# def auto_pipeline(url: str, table_hint: str = None):
#     # ─── 1) Ingest & stage raw data ────────────────────────────────────────────
#     meta = ingest(url)
#     raw_summary = stage_to_snowflake.invoke({
#         "df": meta["df"],
#         "table_hint": table_hint
#     })
#     print(raw_summary)

#     # ─── 2) In‐memory TRANSFORM ────────────────────────────────────────────────
#     df = meta["df"]

#     # 2.1) Drop nulls
#     before = len(df)
#     df = df.dropna()
    
#     print(f"Dropped rows with nulls: {before - len(df)} removed → {len(df)} rows remain")

#     # 2.2) Clean & rename columns (strip spaces/quotes/parens → snake_case)
#     def clean_col(c: str) -> str:
#         c = c.strip(' "')
#         c = re.sub(r'[\(\)]', '', c)
#         c = re.sub(r'[^0-9A-Za-z]+', '_', c).strip('_').lower()
#         return c

#     old_cols = df.columns.tolist()
#     df.columns = [clean_col(c) for c in old_cols]
#     df.columns = [c.upper() for c in df.columns]
#     print(f"Renamed columns:\n  {old_cols!r}\n→ {df.columns.tolist()!r}")
# # … after df.columns = [c.upper() …] …

#     # If you have an INDEX column, drop it (it’s just a row number)
#     # if "INDEX" in df.columns:
#     #     df = df.drop(columns=["INDEX"])
#     #     print("Dropped 'INDEX' column before staging.")

#     # 2.3) Normalize categorical columns into dimension tables
#     dim_tables = {}
#     cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
#     for col in cat_cols:
#         dim_df = df[[col]].drop_duplicates().reset_index(drop=True)
#         hint = f"{table_hint}_{col}_dim" if table_hint else f"{col}_dim"
#         dim_summary = stage_to_snowflake.invoke({"df": dim_df, "table_hint": hint})
#         m = re.search(r" to (\S+)\s*\(", dim_summary)
#         dim_tbl = m.group(1) if m else None
#         dim_tables[col] = dim_tbl
#         print(f"Staged dimension '{col}' → {dim_tbl}")

#         # merge back a numeric key
#         key_df = dim_df.reset_index().rename(columns={"index": f"{col}_id"})
#         df = df.merge(key_df, on=col).drop(columns=[col])

#     from datetime import datetime
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     fact_hint = f"{table_hint}_fact_{ts}" if table_hint else f"fact_{ts}"

#     # Stage the final fact table with the unique hint
#     fact_summary = stage_to_snowflake.invoke({
#         "df": df,
#         "table_hint": fact_hint
#     })
#     print(f"Staged fact table → {fact_summary}")

#     # Persist the new table name for downstream analysis
#     m = re.search(r" to (\S+)\s*\(", fact_summary)
#     fact_table = m.group(1) if m else None
#     Path(".last_table").write_text(fact_table)

#     # 2.4) Stage the final fact table
#     # fact_hint = f"{table_hint}_fact" if table_hint else "fact"
#     # fact_summary = stage_to_snowflake.invoke({"df": df, "table_hint": fact_hint})
#     # m = re.search(r" to (\S+)\s*\(", fact_summary)
#     # fact_table = m.group(1) if m else None
#     # print(f"Staged fact table → {fact_table}")

#     # # Persist the fact‐table name for run_analysis
#     # Path(".last_table").write_text(fact_table)

#     # ─── 3) Suggest analyses on cleaned schema ────────────────────────────────
#     schema = _infer_schema(df)
#     schema_str = ", ".join(f"{c['name']}({c['type']})" for c in schema)

#     llm = ChatOllama(model="llama3:8b", temperature=0.3)
#     prompt_suggest = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a senior data scientist. "
#          f"The cleaned schema is: {schema_str}.  "
#          "List 12 insightful analyses the user could run next. "
#          "Return each idea as '- idea' on its own line."),
#         ("human", "")
#     ])
#     raw_ideas = llm.invoke(prompt_suggest.format()).content.strip()
#     ideas = [line[2:].strip() for line in raw_ideas.splitlines() if line.startswith("-")]
#     print("\nGenerated analysis ideas:")
#     for i, idea in enumerate(ideas, 1):
#         print(f"{i}. {idea}")

#     # ─── 4) Run each analysis and collect chart paths ─────────────────────────
#     chart_info: list[tuple[str,str]] = []
#     for idea in ideas:
#         print(f"\nRunning: {idea}")
#         out = run_analysis.invoke({"request": idea})
#         chart_info.append((idea, out))
#         print(f" → {out}")

#     # ─── 5) Tell a story weaving all charts together ─────────────────────────
#     prompt_story = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a data storyteller.  You ran these analyses:\n"
#          + "\n".join(f"- {idea}: {path}" for idea, path in chart_info)
#          + "\n\nWrite a concise narrative that ties these insights together."),
#         ("human", "")
#     ])
#     story = llm.invoke(prompt_story.format()).content.strip()

#     # ─── 6) Output results ───────────────────────────────────────────────────
#     print("\n=== Narrative Story ===")
#     print(story)
#     Path("story.txt").write_text(story)

#     print("\nCharts saved at:")
#     for _, path in chart_info:
#         print(f" - {path}")

# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("--url",  type=str,  required=True)
#     p.add_argument("--hint", type=str,  dest="table_hint")
#     args = p.parse_args()

#     auto_pipeline(args.url, args.table_hint)

# auto_story_pipeline.py

# import re
# from pathlib import Path
# from datetime import datetime

# import pandas as pd
# from ingestion_engine import ingest
# from staging_storage import stage_to_snowflake
# from analysis_suggester import suggest_analyses
# from analysis_runner import run_analysis
# from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate

# def _infer_schema(df: pd.DataFrame) -> list[dict]:
#     """Infer simple JSON schema from a pandas DataFrame."""
#     type_map = {
#         "int64": "INTEGER",
#         "float64": "FLOAT",
#         "object": "STRING",
#         "bool": "BOOLEAN",
#         "datetime64[ns]": "TIMESTAMP",
#     }
#     return [{"name": n, "type": type_map.get(str(t), "STRING")}
#             for n, t in df.dtypes.items()]

# def auto_pipeline(url: str, table_hint: str = None):
#     # 1) Ingest & stage raw data
#     meta = ingest(url)
#     raw_summary = stage_to_snowflake.invoke({
#         "df": meta["df"],
#         "table_hint": table_hint
#     })
#     print(raw_summary)

#     # 2) In-memory TRANSFORM
#     df = meta["df"]
#     before = len(df)
#     df = df.dropna()
#     print(f"Dropped nulls: {before - len(df)} → {len(df)} rows")

#     # Clean & rename cols
#     def clean_col(c): 
#         c = c.strip(' "'); c = re.sub(r'[\(\)]', '', c)
#         c = re.sub(r'[^0-9A-Za-z]+','_',c).strip('_').lower()
#         return c
#     old = df.columns.tolist()
#     df.columns = [clean_col(c).upper() for c in old]
#     print(f"Renamed columns:\n  {old!r}\n→ {df.columns.tolist()!r}")

#     # Normalize cats into dims
#     for col in df.select_dtypes(include=["object"]).columns:
#         dim_df = df[[col]].drop_duplicates().reset_index(drop=True)
#         hint = f"{table_hint}_{col}_dim" if table_hint else f"{col}_dim"
#         ds = stage_to_snowflake.invoke({"df": dim_df, "table_hint": hint})
#         tbl = re.search(r" to (\S+)\s*\(", ds).group(1)
#         print(f"Staged dim {col} → {tbl}")
#         key_df = dim_df.reset_index().rename(columns={"index":f"{col}_id"})
#         df = df.merge(key_df, on=col).drop(columns=[col])

#     # Stage fact table
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     fact_hint = f"{table_hint}_fact_{ts}" if table_hint else f"fact_{ts}"
#     fact_summary = stage_to_snowflake.invoke({
#         "df": df,
#         "table_hint": fact_hint
#     })
#     print(f"Staged fact table → {fact_summary}")
#     tbl = re.search(r" to (\S+)\s*\(", fact_summary).group(1)
#     Path(".last_table").write_text(tbl)

#     # 3) Suggest analyses
#     schema = _infer_schema(df)
#     schema_str = ", ".join(f"{c['name']}({c['type']})" for c in schema)
#     llm = ChatOllama(model="llama3:8b", temperature=0.3)
#     ps = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a senior data scientist.  "
#          f"Clean schema: {schema_str}.  "
#          "List 12 analyses (one per line as '- idea')."),
#         ("human", "")
#     ])
#     raw_ideas = llm.invoke(ps.format()).content.strip()
#     ideas = [l[2:].strip() for l in raw_ideas.splitlines() if l.startswith("-")]
#     print("\nIdeas:")
#     for i, idea in enumerate(ideas,1):
#         print(f"{i}. {idea}")

#     # 4) Run each analysis, collect HTML + JSON info
#     chart_info = []
#     for idea in ideas:
#         print(f"\nRunning: {idea}")
#         html_path = run_analysis.invoke({"request": idea})
#         chart_info.append((idea, html_path))
#         print(f" → HTML: {html_path}")

#     # ─── 5) Generate per‐chart descriptions from JSON ─────────────────────────
#     desc_dir = Path("descriptions")
#     desc_dir.mkdir(exist_ok=True)
#     desc_list = []

#     for idea, html in chart_info:
#         # sanitize the idea into a key (don’t truncate!)
#         key = re.sub(r'[^0-9A-Za-z]+', '_', idea)

#         # look for any JSON file containing that key
#         candidates = list(Path("charts2/json").glob(f"*{key}*.json"))
#         if not candidates:
#             print(f"[WARN] no JSON found for idea '{idea}', skipping description")
#             continue

#         jsn_path = candidates[0]
#         fig_json = jsn_path.read_text()

#         # save the prompt we send to the LLM
#         prompt = (
#             f"### Chart: {idea}\n"
#             f"Here is the Plotly JSON for that chart:\n```json\n{fig_json}\n```"
#         )
#         (desc_dir / f"{key}_prompt.txt").write_text(prompt)

#         # ask the LLM for a natural‐language description
#         desc = llm.invoke(prompt).content.strip()
#         (desc_dir / f"{key}_description.txt").write_text(desc)

#         desc_list.append(f"- **{idea}**: {desc}")


#     # 6) Final narrative
#     story_prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a data storyteller.  Here are chart insights:\n"
#          + "\n".join(desc_list)
#          + "\n\nWrite a concise narrative tying them together."),
#         ("human", "")
#     ])
#     story = llm.invoke(story_prompt.format()).content.strip()
#     print("\n=== Narrative ===\n", story)
#     Path("story.txt").write_text(story)

# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("--url",  type=str, required=True)
#     p.add_argument("--hint", type=str, dest="table_hint")
#     args = p.parse_args()
#     auto_pipeline(args.url, args.table_hint)

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
