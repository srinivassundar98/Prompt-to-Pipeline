import plotly.express as px
import pandas as pd
fig = px.pnetwork(
    nodes=df[['INDUSTRY_NAME_NZSIOC_ID', 'INDUSTRY_CODE_ANZSIC06_ID']].stack().reset_index(name='node'),
    links=df.groupby('INDUSTRY_AGGREGATION_NZSIOC_ID')['INDUSTRY_NAME_NZSIOC_ID'].apply(list).reset_index(name='source').merge(df.groupby('INDUSTRY_AGGREGATION_NZSIOC_ID')['INDUSTRY_NAME_NZSIOC_ID'].apply(list).reset_index(name='target'), on='INDUSTRY_NAME_NZSIOC_ID', how='cross').drop_duplicates(subset=['source', 'target'], keep='first')