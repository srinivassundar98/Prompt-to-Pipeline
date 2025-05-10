import plotly.express as px
fig = px.line(df.groupby('INDUSTRY_CODE_ANZSIC06_ID')['YEAR'].max().reset_index(name='MAX_YEAR'), x='INDUSTRY_CODE_ANZSIC06_ID', y='MAX_YEAR')