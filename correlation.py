import pandas as pd

df = pd.read_csv('student-mat.csv', sep=';')  # Or student-por.csv
df.head()
df['pass'] = df['G3'] >= 10
df['pass'] = df['pass'].astype(int)
# this will display the first 10 rows in nice tabular form
df[['G3', 'pass']].head(5)
df_encoded = pd.get_dummies(df.drop(columns=['G1', 'G2', 'G3']), drop_first=True)
df_encoded['pass'] = df['pass']
print(df_encoded.head())
# one-hot encode all categorical variables
df_enc = pd.get_dummies(df.drop(columns=['G3']), drop_first=True)

# add the target or any other numeric column back in
df_enc['G3'] = df['G3']

# now compute correlations
corr = df_enc.corr()
print(corr.head())
