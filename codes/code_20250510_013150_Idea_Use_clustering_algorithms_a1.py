import plotly.express as px
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
industry_codes = df['INDUSTRY_CODE_NZSIOC_ID'].values.reshape(-1, 1)
fig = px.scatter(df, x='YEAR', y='VALUE_ID', color='CLUSTER')