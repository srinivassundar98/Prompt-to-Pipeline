import plotly.express as px
import pandas as pd
fig = px.scatter(df, x='VARIABLE_CATEGORY_ID', y='VALUE_ID', color='INDUSTRY_NAME_NZSIOC_ID')