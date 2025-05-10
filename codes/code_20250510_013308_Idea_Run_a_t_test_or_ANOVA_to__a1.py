import plotly.express as px
import pandas as pd
from scipy.stats import ttest_ind, f_oneway
industry_codes = df['INDUSTRY_CODE_ANZSIC06_ID'].unique()
means = []
    industry_df = df[df['INDUSTRY_CODE_ANZSIC06_ID'] == code]
fig = px.box(x=industry_codes, y=means)