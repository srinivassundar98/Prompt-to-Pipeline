import pandas as pd
import plotly.express as px
df = pd.read_csv('data.csv')
fig = px.line(df, x='YEAR', y='VALUE_ID', color='INDUSTRY_CODE_ANZSIC06_ID',
              lineGroup='INDUSTRY_CODE_ANZSIC06_ID', title='Percentage Change in Values Over Time')