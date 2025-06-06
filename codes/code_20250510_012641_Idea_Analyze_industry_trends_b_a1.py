import plotly.express as px
import pandas as pd
df = pd.read_csv('data.csv')  # replace with your data file path
fig = px.line_group(df, x='YEAR', y='VALUE_ID', color='INDUSTRY_CATEGORY_NZSIOC_ID',
                     category_orders={'INDUSTRY_CATEGORY_NZSIOC_ID': df['INDUSTRY_CATEGORY_NZSIOC_ID'].unique()})