import plotly.express as px
import pandas as pd
df = pd.DataFrame({
fig = px.bar(df, x='INDUSTRY_NAME_NZSIOC_ID', y='VALUE_ID',
             color='VARIABLE_CATEGORY_ID', barmode='group')