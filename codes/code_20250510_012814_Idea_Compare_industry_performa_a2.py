import plotly.express as px
import pandas as pd
df = pd.read_csv('data.csv')  # replace with your data file path
fig = px.bar(df, x='INDUSTRY_NAME_NZSIOC_ID', y='VALUE_ID',
             color='VARIABLE_CATEGORY_ID', barmode='group')