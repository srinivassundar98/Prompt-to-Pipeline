import plotly.express as px
fig = px.histogram(df, x='INDUSTRY_AGGREGATION_NZSIOC_ID', color='INDUSTRY_NAME_NZSIOC_ID')