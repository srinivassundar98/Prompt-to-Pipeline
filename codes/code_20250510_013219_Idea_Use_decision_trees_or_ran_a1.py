import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
fig = px.bar(
    x=rfecv.estimators_[rfecv.support_].indices_.values.flatten(),
    y=df['YEAR'],
    color='INDUSTRY_NAME_NZSIOC_ID',
    barmode='group'