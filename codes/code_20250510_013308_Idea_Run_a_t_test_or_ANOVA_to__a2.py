import plotly.express as px
fig = px.box(df, x='INDUSTRY_CODE_ANZSIC06_ID', y='VALUE_ID')