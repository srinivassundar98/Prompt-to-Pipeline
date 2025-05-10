import plotly.express as px
import pandas as pd
df = pd.read_csv('data.csv')
fig = px.box(df, x='INDUSTRY_AGGREGATION_NZSIOC_ID', y='VALUE_ID', color='INDUSTRY_NAME_NZSIOC_ID')