import plotly.express as px
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
fig = px.line(kmf.fit(df['YEAR'], event_observed=df['VALUE_ID']).survival_function_, x='YEAR', y='survival_function_estimate')