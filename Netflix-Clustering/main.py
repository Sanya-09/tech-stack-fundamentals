import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import re

df = pd.read_csv('netflix_titles.csv')  
df.head()

# SELECT RELEVENT FEATURE
features = df[['Genre', 'rating', 'duration']]

features = features.dropna()

print(features)

# Define which columns are categorical and which are numerical
categorical_features = ['Genre', 'rating']
numerical_features = ['Duration']

#Clean duration data 
def parse_duration(x):
    x = str(x)
    if 'min' in x:
        return int(re.search(r'(\d+)', x).group(1))
    elif 'Season' in x or 'Seasons' in x:
        num_seasons = int(re.search(r'(\d+)', x).group(1))
        return num_seasons * 60 * 5  # Example: 5 episodes of 60 min each per season
    else:
        return np.nan

df['duration_min'] = df['duration'].apply(parse_duration)

# Split and explode genres
df['Genre_list'] = df['Genre'].str.split(', ')
all_genres = set([genre for sublist in df['Genre_list'] for genre in sublist])

# One-hot encode genres
for genre in all_genres:
    df[f'genre_{genre}'] = df['Genre_list'].apply(lambda x: int(genre in x))

# Fill any missing ratings
df['rating'] = df['rating'].fillna('Unknown')

# One-hot encode
ohe = OneHotEncoder(sparse_output=False)
rating_encoded = ohe.fit_transform(df[['rating']])
rating_df = pd.DataFrame(rating_encoded, columns=[f'rating_{c}' for c in ohe.categories_[0]])

# Select numerical duration
duration_df = df[['duration_min']].fillna(df['duration_min'].mean())

# Select genre columns
genre_cols = [col for col in df.columns if col.startswith('genre_')]

# Concatenate
X = pd.concat([duration_df.reset_index(drop=True), df[genre_cols].reset_index(drop=True), rating_df.reset_index(drop=True)], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

df['Cluster'] = kmeans.labels_

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='duration_min', y='Cluster', hue='Cluster', palette='Set2')
plt.title("Netflix Shows Clustered by Duration")
plt.show()

# Plot clusters in Rating vs Duration space, colored by cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='rating', y='duration_min', hue='Cluster', palette='Set2')
plt.title('Netflix Shows Clustering')
plt.xlabel('Rating')
plt.ylabel('Duration (minutes)')
plt.show()

# Cluster summary using only numeric columns
cluster_summary = df.groupby('Cluster')[['duration_min']].mean()
print(cluster_summary)

# Count ratings per cluster
rating_counts = df.groupby(['Cluster', 'rating']).size().unstack(fill_value=0)
print(rating_counts)

# Count genres per cluster
df_exploded = df.explode('Genre_list')
genre_counts = df_exploded.groupby(['Cluster', 'Genre_list']).size().unstack(fill_value=0)
print(genre_counts)
