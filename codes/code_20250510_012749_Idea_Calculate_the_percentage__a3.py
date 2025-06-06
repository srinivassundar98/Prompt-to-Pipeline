import pandas as pd
import plotly.express as px
df = pd.read_csv('data.csv')
fig = px.line(df, x='YEAR', y=df.groupby(['INDUSTRY_CODE_ANZSIC06_ID'])['VALUE_ID'].transform(lambda x: (x - x.shift()) / x.shift()).reset_index(0, 'YEAR').drop('YEAR', 1), color='INDUSTRY_CODE_ANZSIC06_ID', title='Percentage Change in Values Over Time')