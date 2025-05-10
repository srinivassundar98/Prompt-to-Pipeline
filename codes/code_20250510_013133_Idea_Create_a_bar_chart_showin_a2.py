import plotly.express as px
fig = px.bar(df.groupby('INDUSTRY_CODE_NZSIOC_ID')['VALUE_ID'].count().reset_index(name='Count').nlargest(10, columns=['Count']), x='INDUSTRY_CODE_NZSIOC_ID', y='Count')