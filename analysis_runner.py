# # analysis_runner.py
# from __future__ import annotations
# import os, tempfile, textwrap
# import pandas as pd
# import plotly.express as px
# from snowflake.connector import connect
# from langchain.tools import tool
# from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate
# from pathlib import Path        # ← add this line

# # 👉 helper to pull a table into Pandas
# def _load_table(table: str) -> pd.DataFrame:
#     conn = connect(
#         user=os.getenv("SNOW_USER"),
#         password=os.getenv("SNOW_PWD"),
#         account=os.getenv("SNOW_ACCOUNT"),
#         role="ACCOUNTADMIN",
#         warehouse="COMPUTE_WH",
#         database="DEMO_DB",
#         schema="RAW",
#     )
#     return pd.read_sql(f"SELECT * FROM {table}", conn)

# @tool
# def run_analysis(request: str) -> str:
#     """
#     Generate Plotly chart based on request for the last‑ingested table.
#     """
#     last_table_path = Path(".last_table")
#     if not last_table_path.exists():
#         return "No ingested table found. Please ingest a dataset first."

#     tbl = last_table_path.read_text().strip()
#     df = _load_table(tbl)

#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a Python data‑viz expert. "
#          "Given a pandas DataFrame called df and a user request, "
#          "write a single Plotly Express snippet (no imports), "
#          "assigning it to variable 'fig'. Do NOT call fig.show()."),
#         ("human", "{request}")
#     ])
#     llm = ChatOllama(model="llama3:8b", temperature=0)
#     code = llm.invoke(prompt.format(request=request)).content.strip()

#     local_vars = {"df": df, "px": px}
#     exec(textwrap.dedent(code), {}, local_vars)

#     fig = local_vars["fig"]
#     tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#     fig.write_image(tmp_png, engine="kaleido")
#     return f"Chart saved to {tmp_png}"

# analysis_runner.py

# from __future__ import annotations
# import os, tempfile, textwrap
# import pandas as pd
# import plotly.express as px
# from pathlib import Path
# from langchain.tools import tool
# from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate

# # Reuse the same Snowflake connection logic as staging_storage
# from staging_storage import _get_connection

# # 👉 helper to pull a table into Pandas
# def _load_table(table: str) -> pd.DataFrame:
#     # Use the demo Snowflake credentials from staging_storage
#     conn = _get_connection()
#     return pd.read_sql(f"SELECT * FROM {table}", conn)

# @tool
# def run_analysis(request: str) -> str:
#     """
#     Generate a Plotly chart based on *request* for the last-ingested table.
#     Returns the PNG file path.
#     """
#     try:
#         tbl = Path(".last_table").read_text().strip()
#     except FileNotFoundError:
#         return "No ingested table found this session."

#     df = _load_table(tbl)        # now uses staging_storage._get_connection

#     # 2) Ask LLM for a short Plotly Express snippet
#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a Python data-viz expert. "
#          "Given a pandas DataFrame called df and a user request, "
#          "write a **single** Plotly Express snippet (no imports) "
#          "that builds the requested chart and assigns it to variable 'fig'. "
#          "Do NOT call fig.show(). Return *only* code."),
#         ("human", "{request}")
#     ])
#     llm   = ChatOllama(model="llama3:8b", temperature=0)
#     code  = llm.invoke(prompt.format(request=request)).content.strip()

#     # 3) Execute code safely
#     local_vars = {"df": df, "px": px}
#     exec(textwrap.dedent(code), {}, local_vars)   # produces 'fig'

#     fig = local_vars["fig"]
#     tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#     fig.write_image(tmp_png, engine="kaleido")
#     return f"Chart saved to {tmp_png}"

# Updated analysis_runner.py with markdown fence stripping

# Updated analysis_runner.py with fuzzy column-name mapping

# Updated analysis_runner.py: print the generated code snippet for debugging

# Updated analysis_runner.py with improved column matching logic
# Updated analysis_runner.py with deeper debug logging and fallback image saving

# Updated run_analysis in analysis_runner.py: always use fig.to_image for saving


#########THIS VERSION WORKS
# from __future__ import annotations
# import os
# import textwrap
# import re
# from difflib import get_close_matches
# from datetime import datetime
# import pandas as pd
# import plotly.express as px
# from pathlib import Path
# from langchain.tools import tool
# from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate

# from staging_storage import _get_connection

# def _load_table(table: str) -> pd.DataFrame:
#     conn = _get_connection()
#     cur = conn.cursor()
#     cur.execute(f"SELECT * FROM {table}")
#     rows = cur.fetchall()
#     cols = [desc[0] for desc in cur.description]
#     return pd.DataFrame(rows, columns=cols)

# @tool
# def run_analysis(request: str) -> str:
#     """
#     Generate a Plotly chart based on *request* for the last-ingested table.
#     Saves the chart via fig.to_image (avoiding fig.write_image hangs).
#     """
#     # Load last table
#     try:
#         tbl = Path(".last_table").read_text().strip()
#     except FileNotFoundError:
#         return "No ingested table found this session."

#     # Load DataFrame
#     df = _load_table(tbl)
#     print(f"[DEBUG] Loaded DataFrame with shape {df.shape}", flush=True)

#     # Ask LLM for code
#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a Python data-viz expert. "
#          "Given a pandas DataFrame called df and a user request, "
#          "write a single Plotly Express snippet (no imports) "
#          "that builds the requested chart and assigns it to variable 'fig'. "
#          "Do NOT call fig.show(). Return only the raw Python code, no markdown fences."),
#         ("human", "{request}")
#     ])
#     llm = ChatOllama(model="llama3:8b", temperature=0)
#     response = llm.invoke(prompt.format(request=request)).content.strip()

#     # Strip fences and debug-print
#     code = re.sub(r"^```(?:python)?\n", "", response)
#     code = re.sub(r"\n```$", "", code).strip()
#     print("[DEBUG] LLM-generated code:\n", code, flush=True)

#     # Normalize column names
#     cols = list(df.columns)
#     norm_map = { re.sub(r'[^0-9a-zA-Z]', '', c).lower(): c for c in cols }
#     def replace_literal(m):
#         key = re.sub(r'[^0-9a-zA-Z]', '', m.group(1)).lower()
#         if key in norm_map: return f"'{norm_map[key]}'"
#         for nk, orig in norm_map.items():
#             if key in nk: return f"'{orig}'"
#         close = get_close_matches(key, norm_map.keys(), n=1, cutoff=0.4)
#         if close: return f"'{norm_map[close[0]]}'"
#         return m.group(0)
#     code = re.sub(r"'([^']+)'", replace_literal, code)
#     print("[DEBUG] Post-mapping code:\n", code, flush=True)

#     # Execute and get fig
#     local_vars = {"df": df, "px": px}
#     print("[DEBUG] Executing code snippet...", flush=True)
#     exec(textwrap.dedent(code), {}, local_vars)
#     fig = local_vars.get("fig")
#     if fig is None:
#         return "Failed to generate figure."
#     print("[DEBUG] Figure object created", flush=True)
# # inside run_analysis, after fig is created:

#     out_dir = Path("charts")
#     out_dir.mkdir(parents=True, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     safe_req = re.sub(r'\W+', '_', request).strip('_')[:50]

#     html_file = out_dir / f"chart_{timestamp}_{safe_req}.html"
#     fig.write_html(str(html_file), include_plotlyjs="cdn")
#     print(f"[DEBUG] write_html succeeded: {html_file}", flush=True)
#     return f"Chart saved to {html_file}"


# After saving this function in analysis_runner.py, rerun:
# python -u graph_pipeline.py --analysis "correlation between height and weight"

# analysis_runner.py

# Updated analysis_runner.py

# Updated analysis_runner.py with deterministic regression handling

# Updated `analysis_runner.py` with improved LLM‐only snippet extraction
# Updated analysis_runner.py with additional debug statements in run_analysis
# analysis_runner.py

# analysis_runner.py

# analysis_runner.py
############CHECKKKK HEREEEEE PLZ############3
# from __future__ import annotations
# import textwrap
# import re
# from pathlib import Path
# from datetime import datetime
# import pandas as pd
# import plotly.express as px
# from langchain.tools import tool
# from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate
# import statsmodels as sm
# from staging_storage import _get_connection

# def _load_table(table: str) -> pd.DataFrame:
#     """
#     Load the specified Snowflake table into a DataFrame.
#     """
#     conn = _get_connection()
#     cur = conn.cursor()
#     cur.execute(f"SELECT * FROM {table}")
#     rows = cur.fetchall()
#     cols = [desc[0] for desc in cur.description]
#     return pd.DataFrame(rows, columns=cols)

# @tool
# def run_analysis(request: str) -> str:
#     """
#     Run an analysis by asking the LLM to generate a Plotly Express snippet.
#     This version prints detailed debug information, including:
#       - The raw LLM response (thoughts)
#       - The extracted code snippet
#       - Execution progress
#     """
#     print("[DEBUG] run_analysis called", flush=True)
#     print(f"[DEBUG] User request: {request}", flush=True)

#     # 1) Read the last table name
#     tbl = Path(".last_table").read_text().strip()
#     print(f"[DEBUG] Last table: {tbl}", flush=True)

#     # 2) Load the DataFrame
#     df = _load_table(tbl)
#     df_f=df.copy()
#     # df_f=df.rename(columns={'Index': 'Index', ' Height(Inches)"': 'Height', ' "Weight(Pounds)"': 'Weight'})
#     print(f"[DEBUG] DataFrame columns: {list(df_f.columns)}", flush=True)
#     print(f"[DEBUG] DataFrame preview:\n{df_f.head()}", flush=True)

#     # 3) Build the prompt with actual column names
#     cols = list(df_f.columns)
#     system_message = (
#         "You are a Python data-viz expert.  "
#         f"The DataFrame `df` has columns: {cols!r}.  "
#         "Produce **only** a Plotly Express snippet (no imports) "
#         "that assigns the chart to variable `fig`.  "
#         "If the request mentions regression, include `trendline='ols'`.  "
#         "Return exactly the code block—no explanation, no markdown fences, no backticks."
#     )

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_message),
#         ("human", "{request}")
#     ])
#     llm = ChatOllama(model="llama3:8b", temperature=0)

#     # 4) Invoke the LLM
#     print("[DEBUG] Sending prompt to LLM...", flush=True)
#     raw = llm.invoke(prompt.format(request=request)).content
#     print(f"[DEBUG] Raw LLM response (thoughts):\n{raw}", flush=True)
#     # 1) Split into lines and drop any backtick‐fence lines
#     lines = raw.splitlines()
#     clean_lines = [ln for ln in lines if not re.match(r'^\s*```', ln)]

#     # 2) Find the first actual code line
#     start_idx = 0
#     for i, ln in enumerate(clean_lines):
#         if ln.strip().startswith(("import ", "fig", "px")):
#             start_idx = i
#             break

#     # 3) Reconstruct the cleaned code block
#     code = "\n".join(clean_lines[start_idx:]).strip()
#     print("[DEBUG] Cleaned code for execution:\n", code, flush=True)

#     # 4) Execute the cleaned snippet
#     local_ns = {"df": df_f, "px": px}
#     exec(textwrap.dedent(code), {}, local_ns)
#     fig = local_ns.get("fig")

#     # # 5) Extract the code snippet
#     # m = re.search(r"```(?:python)?\n([\s\S]+?)\n```", raw)
#     # if m:
#     #     code = m.group(1).strip()
#     #     print("[DEBUG] Extracted code from fences:", flush=True)
#     # else:
#     #     code = raw.strip()
#     #     print("[DEBUG] No fences found; using entire response as code:", flush=True)
#     # print(code, flush=True)

#     # # 6) Execute the snippet
#     # print("[DEBUG] Executing code snippet...", flush=True)
#     # local_ns: dict = {"df": df_f, "px": px}
#     # exec(textwrap.dedent(code), {}, local_ns)
#     # fig = local_ns.get("fig")
#     if fig is None:
#         return f"Failed to generate figure. Executed code was:\n{code}"

# # 7) Save the figure as HTML
#     charts_dir = Path("charts3")
#     charts_dir.mkdir(exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     safe = re.sub(r'\W+', '_', request)[:30]
#     out_path = charts_dir / f"chart_{ts}_{safe}.html"

#     print(f"[DEBUG] Saving chart HTML to {out_path}", flush=True)
#     fig.write_html(str(out_path), include_plotlyjs="cdn")
#     print("[DEBUG] Chart HTML saved successfully", flush=True)

#     return f"Chart saved to {out_path}"
# Here's the fully revised `analysis_runner.py` with the enhanced `run_analysis` tool.

# import os
# from pathlib import Path
# import pandas as pd
# import plotly.express as px
# import textwrap
# import traceback
# import re
# from datetime import datetime
# from snowflake.connector import connect
# from staging_storage import _get_connection

# from langchain.tools import tool
# from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate

# def _load_table(table: str) -> pd.DataFrame:
#     """
#     Load the given Snowflake table into a pandas DataFrame.
#     Expects .last_table to have been written with the raw or fact table name.
#     """
#     conn = _get_connection()
#     return pd.read_sql(f"SELECT * FROM {table}", conn)

# def save_fig_as_html(fig, request: str) -> str:
#     """
#     Save a Plotly figure to an HTML file under charts2/ with a timestamped name.
#     Returns the filesystem path as a string.
#     """
#     charts_dir = Path("charts2")
#     charts_dir.mkdir(exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     safe = re.sub(r'[^0-9A-Za-z]+', '_', request)[:30]
#     out_path = charts_dir / f"chart_{ts}_{safe}.html"
#     fig.write_html(str(out_path), include_plotlyjs="cdn")
#     return str(out_path)

# @tool
# def run_analysis(request: str, previous_ideas: str | None = None) -> str:
#     """
#     Generate a Plotly chart based on *request*. On error, feed back to the LLM
#     and retry up to 3 times. Returns the HTML path or a final error message.
#     """
#     # Load the last‐ingested table name
#     try:
#         tbl = Path(".last_table").read_text().strip()
#     except FileNotFoundError:
#         return "No ingested table found this session."

#     # Load the DataFrame
#     df = _load_table(tbl)

#     # Build schema string for the prompt
#     schema_str = ", ".join(f"'{col}'" for col in df.columns)

#     def make_prompt(schema: str,
#                     code_context: str = "",
#                     error_msg: str = "") -> ChatPromptTemplate:
#         """
#         Build a ChatPromptTemplate that:
#           1. Knows the true schema
#           2. Uses only {request} placeholder
#           3. Safely escapes any braces in code or errors
#         """
#         system = (
#             "You are a Python data‑viz expert.  "
#             f"The pandas DataFrame `df` has these exact columns: {schema}.  "
#             "Write a Plotly Express snippet (no imports) assigning the chart to `fig`."
#             "If the request mentions regression, include `trendline='ols'`."
#             "Return exactly the code block—no explanation, no markdown fences, no backticks.\n"
#         )
#         if code_context:
#             cc = code_context.replace("{", "{{").replace("}", "}}")
#             system += f"Your last code was:\n```\n{cc}\n```\n"
#         if error_msg:
#             em = error_msg.replace("{", "{{").replace("}", "}}")
#             system += f"It failed with error:\n```\n{em}\n```\n"
#         system += "Now produce corrected code."

#         return ChatPromptTemplate.from_messages([
#             ("system", system),
#             ("human", "{request}")
#         ])

#     llm = ChatOllama(model="llama3:8b", temperature=0)
#     prompt = make_prompt(schema=schema_str)

#     last_error = ""
#     for attempt in range(1, 4):
#         # 1) Ask for code
#         raw = llm.invoke(prompt.format(request=request)).content.strip()

#         # 2) Extract only code lines
#         lines = []
#         for ln in raw.splitlines():
#             if re.match(r'^\s*```', ln):
#                 continue
#             if re.match(r'^\s*(#|import |from |fig|px|\w+\s*=)', ln):
#                 lines.append(ln)
#         start = next((i for i, ln in enumerate(lines)
#                       if re.match(r'^\s*(import |fig|px|\w+\s*=)', ln)), 0)
#         code = "\n".join(lines[start:]).strip()
#         # Remove known-bad kwargs
#         code = code.replace("color_discrete=None", "")

#         print(f"[DEBUG] Attempt {attempt}, executing code:\n{code}", flush=True)

#         # 3) Execute
#         try:
#             local_ns = {"df": df, "px": px}
#             exec(textwrap.dedent(code), {}, local_ns)
#             fig = local_ns.get("fig")
#             if not fig:
#                 raise RuntimeError("Code executed but did not set 'fig'.")
#             # 4) Save and return
#             return save_fig_as_html(fig, request)

#         except Exception:
#             last_error = traceback.format_exc()
#             print(f"[DEBUG] Attempt {attempt} failed:\n{last_error}", flush=True)
#             # Prepare next prompt with escaped code & error
#             prompt = make_prompt(
#                 schema=schema_str,
#                 code_context=code,
#                 error_msg=last_error
#             )

#     return f"Failed after 3 attempts. Last error:\n{last_error}"
########### THIS WORKS#######
# import os
# from pathlib import Path
# import pandas as pd
# import plotly.express as px
# import textwrap
# import traceback
# import re
# from datetime import datetime
# from snowflake.connector import connect
# from staging_storage import _get_connection

# from langchain.tools import tool
# from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate

# def _load_table(table: str) -> pd.DataFrame:
#     conn = _get_connection()
#     return pd.read_sql(f"SELECT * FROM {table}", conn)

# def save_fig_as_html(fig, request: str) -> str:
#     charts_dir = Path("charts2")
#     charts_dir.mkdir(exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     safe = re.sub(r'[^0-9A-Za-z]+', '_', request)[:30]
#     out_path = charts_dir / f"chart_{ts}_{safe}.html"
#     fig.write_html(str(out_path), include_plotlyjs="cdn")
#     return str(out_path)

# @tool
# def run_analysis(request: str, previous_ideas: str | None = None) -> str:
#     """
#     Generate a Plotly chart based on the user’s request, retrying up to 3 times
#     by feeding any execution error back into the LLM to correct itself.
#     Returns the path to the saved HTML chart or an error message.
#     """
#     # Load last table
#     try:
#         tbl = Path(".last_table").read_text().strip()
#     except FileNotFoundError:
#         return "No ingested table found this session."
#     df = _load_table(tbl)

#     # Build schema string
#     schema_str = ", ".join(f"'{col}'" for col in df.columns)

#     def make_prompt(schema: str,
#                     code_context: str = "",
#                     error_msg: str = "") -> ChatPromptTemplate:
#         # Base system message, including schema and retry guidance
#         system = (
#             "You are a Python data-viz expert.\n"
#             f"The pandas DataFrame `df` has these exact columns: {schema}.\n"
#             "Write a Plotly Express snippet (no imports) assigning the chart to `fig`.\n"
#         )
#         if code_context:
#             cc = code_context.replace("{", "{{").replace("}", "}}")
#             system += f"Your last code was:\n```\n{cc}\n```\n"
#         if error_msg:
#             em = error_msg.replace("{", "{{").replace("}", "}}")
#             system += f"It failed with error:\n```\n{em}\n```\n"
#         system += (
#             "Because your prior snippet errored, do NOT repeat it verbatim—"
#             "please correct it now."
#         )

#         return ChatPromptTemplate.from_messages([
#             ("system", system),
#             ("human", "{request}")
#         ])

#     llm    = ChatOllama(model="llama3:8b", temperature=0)
#     prompt = make_prompt(schema=schema_str)
#     last_code  = ""
#     last_error = ""

#     for attempt in range(1, 4):
#         print(f"[DEBUG] Prompting LLM (attempt {attempt}):")
# # Render the full formatted prompt and print it
#         pv = prompt.format_prompt(request=request)
#         print("[DEBUG] Full prompt to LLM:\n", pv.to_string(), flush=True)


#         raw = llm.invoke(prompt.format(request=request)).content.strip()

#         # Extract only plausible code lines
#         lines = []
#         for ln in raw.splitlines():
#             if re.match(r'^\s*```', ln): continue
#             if re.match(r'^\s*(#|import |from |fig|px|\w+\s*=)', ln):
#                 lines.append(ln)
#         start = next((i for i,ln in enumerate(lines)
#                       if re.match(r'^\s*(import |fig|px|\w+\s*=)', ln)), 0)
#         code = "\n".join(lines[start:]).strip()
#         code = code.replace("color_discrete=None", "")

#         print(f"[DEBUG] Attempt {attempt}, executing code:\n{code}", flush=True)

#         try:
#             local_ns = {"df": df, "px": px}
#             exec(textwrap.dedent(code), {}, local_ns)
#             fig = local_ns.get("fig")
#             if not fig:
#                 raise RuntimeError("No fig produced.")
#             return save_fig_as_html(fig, request)

#         except Exception:
#             last_error = traceback.format_exc()
#             print(f"[DEBUG] Attempt {attempt} failed:\n{last_error}", flush=True)

#             # Build next prompt, escaping any braces
#             prompt = make_prompt(
#                 schema=schema_str,
#                 code_context=code,
#                 error_msg=last_error
#             )
#             # If the LLM just repeated the same code, nudge harder
#             if code.strip() == last_code.strip():
#                 system = prompt.messages[0].content + (
#                     "\n*You repeated the same snippet; "
#                     "please modify it to fix the error above.*\n"
#                 )
#                 prompt = ChatPromptTemplate.from_messages([
#                     ("system", system),
#                     ("human", "{request}")
#                 ])

#             last_code = code

#     return f"Failed after 3 attempts. Last error:\n{last_error}"

# analysis_runner.py

# import os
# import json
# from pathlib import Path
# import pandas as pd
# import plotly.express as px
# import textwrap
# import traceback
# import re
# from datetime import datetime
# from snowflake.connector import connect
# from staging_storage import _get_connection
# import sklearn
# from langchain.tools import tool
# from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate

# # Ensure dump directories exist
# for d in ("charts2", "charts2/json", "codes", "prompts", "responses"):
#     Path(d).mkdir(exist_ok=True)

# def _load_table(table: str) -> pd.DataFrame:
#     """Load the given Snowflake table into a pandas DataFrame."""
#     conn = _get_connection()
#     return pd.read_sql(f"SELECT * FROM {table}", conn)

# def save_fig_as_html(fig, request: str) -> str:
#     """Save a Plotly figure as HTML under charts2/ and return its path."""
#     charts_dir = Path("charts2")
#     charts_dir.mkdir(exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     safe = re.sub(r'[^0-9A-Za-z]+', '_', request)[:30]
#     out_path = charts_dir / f"chart_{ts}_{safe}.html"
#     fig.write_html(str(out_path), include_plotlyjs="cdn")
#     return str(out_path)

# @tool(description="Generate a Plotly chart based on the user's request, with retry on error.")
# def run_analysis(request: str, previous_ideas: str | None = None) -> str:
#     """
#     Generate a Plotly chart based on *request*, retrying up to 3 times
#     feeding any execution error back into the LLM. Logs prompts, responses,
#     code, and saves Plotly JSON for downstream description.
#     Returns the HTML path (existing behavior).
#     """
#     # 0) load last table
#     try:
#         tbl = Path(".last_table").read_text().strip()
#     except FileNotFoundError:
#         return "No ingested table found this session."
#     df = _load_table(tbl)

#     # Build schema string
#     schema_str = ", ".join(f"'{col}'" for col in df.columns)

#     def make_prompt(schema: str,
#                     code_context: str = "",
#                     error_msg: str = "") -> ChatPromptTemplate:
#         """Construct a ChatPromptTemplate with schema + retry instructions."""
#         system = (
#             "You are a Python data-viz expert.\n"
#             f"The pandas DataFrame `df` has these exact columns: {schema}.\n"
#             "Write a Plotly Express snippet—feel free to include any necessary imports—assigning the chart to `fig`.\n"
#         )
#         if code_context:
#             cc = code_context.replace("{", "{{").replace("}", "}}")
#             system += f"Your last code was:\n```\n{cc}\n```\n"
#         if error_msg:
#             em = error_msg.replace("{", "{{").replace("}", "}}")
#             system += f"It failed with error:\n```\n{em}\n```\n"
#         system += (
#             "Because your prior snippet errored, do NOT repeat it verbatim—"
#             "please correct it now."
#         )
#         return ChatPromptTemplate.from_messages([
#             ("system", system),
#             ("human", "{request}")
#         ])

#     llm    = ChatOllama(model="llama3:8b", temperature=0)
#     prompt = make_prompt(schema=schema_str)
#     last_code  = ""
#     last_error = ""
#     ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
#     safe = re.sub(r'[^0-9A-Za-z]+', '_', request)[:30]

#     for attempt in range(1, 4):
#         # — Dump the formatted prompt
#         prompt_text = prompt.format(request=request)
#         (Path("prompts") / f"prompt_{ts}_{safe}_a{attempt}.txt").write_text(prompt_text)

#         # Invoke LLM
#         raw = llm.invoke(prompt_text).content.strip()
#         # — Dump raw response
#         (Path("responses") / f"response_{ts}_{safe}_a{attempt}.txt").write_text(raw)

#         # Extract code lines
#         lines = []
#         for ln in raw.splitlines():
#             if re.match(r'^\s*```', ln): continue
#             if re.match(r'^\s*(#|import |from |fig|px|\w+\s*=)', ln):
#                 lines.append(ln)
#         start = next((i for i, ln in enumerate(lines)
#                       if re.match(r'^\s*(import |fig|px|\w+\s*=)', ln)), 0)
#         code = "\n".join(lines[start:]).strip()
#         code = code.replace("color_discrete=None", "")
#         # — Dump extracted code
#         (Path("codes") / f"code_{ts}_{safe}_a{attempt}.py").write_text(code)

#         try:
#             print(f"[DEBUG] Attempt {attempt}, executing code:\n{code}", flush=True)
#             local_ns = {"df": df, "px": px}
#             exec(textwrap.dedent(code), {}, local_ns)
#             fig = local_ns.get("fig")
#             if not fig:
#                 raise RuntimeError("No fig produced.")

#             # — Dump Plotly JSON
#             json_path = Path("charts2/json") / f"chart_{ts}_{safe}.json"
#             with open(json_path, "w") as jp:
#                 json.dump(fig.to_plotly_json(), jp)

#             # Save and return HTML
#             html_path = save_fig_as_html(fig, request)
#             print(f"[DEBUG] Saved JSON at {json_path}", flush=True)
#             return html_path

#         except Exception:
#             last_error = traceback.format_exc()
#             print(f"[DEBUG] Attempt {attempt} failed:\n{last_error}", flush=True)
#             # Build next prompt
#             prompt = make_prompt(
#                 schema=schema_str,
#                 code_context=code,
#                 error_msg=last_error
#             )
#             # If repeated snippet, nudge harder
#             if code.strip() == last_code.strip():
#                 system = prompt.messages[0].content + (
#                     "\n*You repeated the same snippet; "
#                     "please modify it to fix the error above.*\n"
#                 )
#                 prompt = ChatPromptTemplate.from_messages([
#                     ("system", system),
#                     ("human", "{request}")
#                 ])
#             last_code = code

#     return f"Failed after 3 attempts. Last error:\n{last_error}"

# analysis_runner.py

# import os
# import json
# import textwrap
# import traceback
# import re
# from datetime import datetime
# from pathlib import Path

# import pandas as pd
# import plotly.express as px
# from snowflake.connector import connect

# from staging_storage import _get_connection
# from langchain.tools import tool
# from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate

# # Ensure dump directories exist
# for d in ("charts2", "charts2/json", "codes", "prompts", "responses"):
#     Path(d).mkdir(exist_ok=True)

# def _load_table(table: str) -> pd.DataFrame:
#     """
#     Load the given Snowflake table into a pandas DataFrame.
#     Expects .last_table to have been written with the raw or fact table name.
#     """
#     conn = _get_connection()
#     return pd.read_sql(f"SELECT * FROM {table}", conn)

# def save_fig_as_html(fig, request: str) -> str:
#     """
#     Save a Plotly figure to an HTML file under charts2/ with a timestamped name.
#     Returns the filesystem path as a string.
#     """
#     charts_dir = Path("charts2")
#     charts_dir.mkdir(exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     safe = re.sub(r'[^0-9A-Za-z]+', '_', request)[:30]
#     out_path = charts_dir / f"chart_{ts}_{safe}.html"
#     fig.write_html(str(out_path), include_plotlyjs="cdn")
#     return str(out_path)

# @tool(description="Generate a Plotly chart based on the user's request, with automatic retry on error.")
# def run_analysis(request: str, previous_ideas: str | None = None) -> str:
#     """
#     Generate a Plotly chart based on *request*, retrying up to 3 times
#     by feeding any execution error back into the LLM to correct itself.
#     Logs prompts, responses, code, and saves Plotly JSON for downstream use.
#     Returns the path to the saved HTML chart or an error message.
#     """
#     # 0) load last table name
#     try:
#         tbl = Path(".last_table").read_text().strip()
#     except FileNotFoundError:
#         return "No ingested table found this session."

#     # 1) load the DataFrame
#     df = _load_table(tbl)

#     # 2) build schema string for prompt
#     schema_str = ", ".join(f"'{col}'" for col in df.columns)

#     def make_prompt(schema: str,
#                     code_context: str = "",
#                     error_msg: str = "") -> ChatPromptTemplate:
#         """
#         Construct a ChatPromptTemplate that:
#          - knows the true schema,
#          - allows imports,
#          - includes prior code+error on retries,
#          - instructs not to repeat the same snippet verbatim.
#         """
#         system = (
#             "You are a Python data-viz expert.\n"
#             f"The pandas DataFrame `df` has these exact columns: {schema}.\n"
#             "Write a Plotly Express snippet (you may include imports) assigning the chart to variable `fig`.\n"
#         )
#         if code_context:
#             cc = code_context.replace("{", "{{").replace("}", "}}")
#             system += f"Your last code was:\n```\n{cc}\n```\n"
#         if error_msg:
#             em = error_msg.replace("{", "{{").replace("}", "}}")
#             system += f"It failed with error:\n```\n{em}\n```\n"
#         system += (
#             "Because your prior snippet errored, do NOT repeat it verbatim—"
#             "please correct it now."
#         )
#         return ChatPromptTemplate.from_messages([
#             ("system", system),
#             ("human", "{request}")
#         ])

#     llm = ChatOllama(model="llama3:8b", temperature=0)
#     prompt = make_prompt(schema=schema_str)
#     last_code = ""
#     last_error = ""
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     safe = re.sub(r'[^0-9A-Za-z]+', '_', request)[:30]

#     # 3) attempt loop
#     for attempt in range(1, 4):
#         # render and log prompt
#         prompt_text = prompt.format(request=request)
#         Path("prompts").joinpath(f"prompt_{ts}_{safe}_a{attempt}.txt").write_text(prompt_text)

#         # invoke LLM and log raw response
#         raw = llm.invoke(prompt_text).content.strip()
#         Path("responses").joinpath(f"response_{ts}_{safe}_a{attempt}.txt").write_text(raw)

#         # extract code lines
#         lines = []
#         for ln in raw.splitlines():
#             if re.match(r'^\s*```', ln):
#                 continue
#             if re.match(r'^\s*(#|import |from |fig|px|\w+\s*=)', ln):
#                 lines.append(ln)
#         start_idx = next((i for i, ln in enumerate(lines)
#                           if re.match(r'^\s*(import |fig|px|\w+\s*=)', ln)), 0)
#         code = "\n".join(lines[start_idx:]).strip()
#         # remove any fig.show() to prevent pop-ups
#         code = code.replace("fig.show()", "").strip()
#         # log the code
#         Path("codes").joinpath(f"code_{ts}_{safe}_a{attempt}.py").write_text(code)

#         print(f"[DEBUG] Attempt {attempt}, executing code:\n{code}", flush=True)
#         try:
#             local_ns = {"df": df, "px": px}
#             exec(textwrap.dedent(code), {}, local_ns)
#             fig = local_ns.get("fig")
#             if not fig:
#                 raise RuntimeError("Code executed but did not produce 'fig'.")

#             # save Plotly JSON
#             json_path = Path("charts2/json") / f"chart_{ts}_{safe}.json"
#             with open(json_path, "w") as jp:
#                 json.dump(fig.to_plotly_json(), jp)

#             # save HTML and return
#             html_path = save_fig_as_html(fig, request)
#             print(f"[DEBUG] Saved JSON at {json_path}", flush=True)
#             return html_path

#         except Exception:
#             last_error = traceback.format_exc()
#             print(f"[DEBUG] Attempt {attempt} failed:\n{last_error}", flush=True)

#             # rebuild prompt for retry
#             prompt = make_prompt(
#                 schema=schema_str,
#                 code_context=code,
#                 error_msg=last_error
#             )
#             # if LLM repeated the same code, add a nudge
#             if code.strip() == last_code.strip():
#                 full_sys = prompt.format_prompt(request=request).to_string()
#                 full_sys += (
#                     "\n*You returned the same snippet; please modify it so it runs without error.*\n"
#                 )
#                 prompt = ChatPromptTemplate.from_messages([
#                     ("system", full_sys),
#                     ("human", "{request}")
#                 ])
#             last_code = code

#     # if we get here, all retries failed
#     return f"Failed after 3 attempts. Last error:\n{last_error}"


##########ADD THIS

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
