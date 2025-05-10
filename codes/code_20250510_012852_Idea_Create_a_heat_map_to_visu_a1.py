import pandas as pd
import plotly.express as px
df = pd.read_csv('data.csv')
fig = px.heatmap(df.pivot_table(index='INDUSTRY_NAME_NZSIOC_ID', columns='VARIABLE_CODE_ID', values='VALUE_ID', aggfunc='mean'), 
                 color_continuous_scale='Blues')