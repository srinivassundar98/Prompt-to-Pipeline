import pandas as pd
import plotly.express as px
fig = px.scatter(df, x='YEAR', y='VALUE_ID', color='INDUSTRY_NAME_NZSIOC_ID')