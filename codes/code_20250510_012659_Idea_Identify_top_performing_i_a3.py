import plotly.express as px
import pandas as pd
df = pd.DataFrame({
fig = px.bar(df.groupby('INDUSTRY_NAME_NZSIOC_ID')['VALUE_ID'].sum().reset_index(name='Total Value'),
             x='INDUSTRY_NAME_NZSIOC_ID', y='Total Value',
             title='Top-Performing Industries by Total Value')