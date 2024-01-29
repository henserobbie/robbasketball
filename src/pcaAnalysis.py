import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# import data
basePath = os.getcwd()
march = pd.read_csv(f'{basePath}/data/processed/marchGames.csv', dtype = float, \
                        converters = {'Season':int, 'TeamID':int})

# remove non input columns
X = march.drop(columns=['Season', 'TeamID1', 'Score1', 'TeamID2', 'Score2'])

# Standardize the features (important for PCA)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

# Calculate the percentage of variance explained
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Plot the percentage of variance explained vs dimensions
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.title('Percentage of Variance Explained vs Dimensions')
plt.xlabel('Number of Dimensions')
plt.ylabel('Cumulative Variance Explained')
plt.grid(True)
plt.show()

# 
pairwise_distances = np.linalg.norm(X - np.mean(X, axis=0), axis=1)

# Plot the distribution of distances
plt.figure(figsize=(10, 6))
sns.histplot(pairwise_distances, kde=True)
plt.title('Distribution of Distances After PCA')
plt.xlabel('Pairwise Distance')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()