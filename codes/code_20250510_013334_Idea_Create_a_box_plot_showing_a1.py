import pandas as pd
import plotly.express as px
fig = px.box(df, x='VARIABLE_CATEGORY_ID', y='VALUE_ID')