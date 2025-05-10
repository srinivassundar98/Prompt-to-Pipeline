
# analysis_runner.py

import os
import json
import textwrap
import traceback
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
from snowflake.connector import connect

from staging_storage import _get_connection
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

# 0) Ensure our dump directories exist
for d in ("charts2", "charts2/json", "codes", "prompts", "responses"):
    Path(d).mkdir(exist_ok=True)

def _load_table(table: str) -> pd.DataFrame:
    """Load the given Snowflake table into a pandas DataFrame."""
    conn = _get_connection()
    return pd.read_sql(f"SELECT * FROM {table}", conn)

def save_fig_as_html(fig, request: str) -> str:
    """Save a Plotly figure as HTML under charts2/ and return its path."""
    charts_dir = Path("charts2")
    charts_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r'[^0-9A-Za-z]+', '_', request)[:30]
    out_path = charts_dir / f"chart_{ts}_{safe}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return str(out_path)

@tool(description="Generate a Plotly chart based on the user's request, retrying on error.")
def run_analysis(request: str, previous_ideas: str | None = None) -> str:
    """
    Generate a Plotly chart based on *request*, retrying up to 3 times
    by feeding any execution error back into the LLM to correct itself.
    Logs prompts, responses, code, and saves Plotly JSON for downstream use.
    Returns the path to the saved HTML chart or an error message.
    """
    # 1) Load the name of the last ingested table
    try:
        tbl = Path(".last_table").read_text().strip()
    except FileNotFoundError:
        return "No ingested table found this session."

    # 2) Pull that table into a DataFrame
    df = _load_table(tbl)

    # 3) Build a string listing the exact columns (for the LLM)
    schema_str = ", ".join(f"'{col}'" for col in df.columns)

    def make_prompt(schema: str,
                    code_context: str = "",
                    error_msg: str = "") -> ChatPromptTemplate:
        """
        Construct a ChatPromptTemplate that
        - knows the true schema,
        - allows imports,
        - includes prior code+error on retries,
        - instructs not to repeat the same snippet verbatim.
        """
        system = (
            "You are a Python data-viz expert.\n"
            f"The pandas DataFrame `df` has these exact columns: {schema}.\n"
            "Do not load any data. That is not needed. The data is already stored in df. Do not try to define/redefine df."
            "Write a Plotly Express snippet without ANY comments (you may include imports) assigning the chart to variable `fig`."
            "Make sure you close all the braces properly."
            "Make sure you omit ALL comments\n"
        )
        if code_context:
            cc = code_context.replace("{", "{{").replace("}", "}}")
            system += f"Your last code was:\n```\n{cc}\n```\n"
        if error_msg:
            em = error_msg.replace("{", "{{").replace("}", "}}")
            system += f"It failed with error:\n```\n{em}\n```\n"
        system += (
            "Because your prior snippet errored, do NOT repeat it verbatim—"
            "please correct it now."
        )
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{request}")
        ])

    llm = ChatOllama(model="llama3:8b", temperature=0)
    prompt = make_prompt(schema=schema_str)
    last_code = ""
    last_error = ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r'[^0-9A-Za-z]+', '_', request)[:30]

    # 4) Try up to 3 times
    for attempt in range(1, 4):
        # --- Render & save the prompt ---
        prompt_text = prompt.format(request=request)
        Path("prompts").joinpath(f"prompt_{ts}_{safe}_a{attempt}.txt").write_text(prompt_text)

        # --- Invoke LLM & save raw response ---
        raw = llm.invoke(prompt_text).content.strip()
        Path("responses").joinpath(f"response_{ts}_{safe}_a{attempt}.txt").write_text(raw)

        # --- Extract the code snippet ---
        lines = []
        for ln in raw.splitlines():
            if re.match(r"^\s*```", ln):
                continue
            if re.match(r"^\s*(#|import |from |fig|px|\w+\s*=)", ln):
                lines.append(ln)
        start_idx = next((i for i, ln in enumerate(lines)
                          if re.match(r"^\s*(import |fig|px|\w+\s*=)", ln)), 0)
        code = "\n".join(lines[start_idx:]).strip()
        # remove any fig.show() so no pop-ups
        code = code.replace("fig.show()", "").strip()

        # --- Save the extracted code ---
        Path("codes").joinpath(f"code_{ts}_{safe}_a{attempt}.py").write_text(code)

        print(f"[DEBUG] Attempt {attempt}, executing code:\n{code}", flush=True)
        try:
            # Execute the snippet
            local_ns = {"df": df, "px": px}
            exec(textwrap.dedent(code), {}, local_ns)
            fig = local_ns.get("fig")
            if not fig:
                raise RuntimeError("Code executed but did not produce 'fig'.")

            # --- Save Plotly JSON ---
            json_path = Path("charts2/json") / f"chart_{ts}_{safe}.json"
            with open(json_path, "w") as jp:
                json.dump(fig.to_plotly_json(), jp)

            # --- Save HTML & return ---
            html_path = save_fig_as_html(fig, request)
            print(f"[DEBUG] Saved JSON at {json_path}", flush=True)
            return html_path

        except Exception:
            last_error = traceback.format_exc()
            print(f"[DEBUG] Attempt {attempt} failed:\n{last_error}", flush=True)

            # --- Rebuild the prompt for retry ---
            prompt = make_prompt(
                schema=schema_str,
                code_context=code,
                error_msg=last_error
            )

            # If the model just repeated the same snippet, force it to change
            if code.strip() == last_code.strip():
                # Escape any braces in the full prompt before re-injecting
                full_sys = prompt.format_prompt(request=request).to_string()
                full_sys = full_sys.replace("{", "{{").replace("}", "}}")
                full_sys += (
                    "\n*You returned the same snippet; please modify it so it runs without error.*\n"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", full_sys),
                    ("human", "{request}")
                ])

            last_code = code

    # 5) If we exhaust retries, return the error string
    return f"Failed after 3 attempts. Last error:\n{last_error}"
