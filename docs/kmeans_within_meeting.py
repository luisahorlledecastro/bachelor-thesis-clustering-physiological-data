"""
k_means.py
Author: Luísa Hörlle de Castro
"""

# import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import mutual_info_score, adjusted_rand_score
from copy import deepcopy
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from sklearn.decomposition import PCA

# import modules
from prep.scr import SCR


class KWithin:
    def __init__(self):
        # initialise prepped data
        self.scr = SCR()
        self.all_clusters = None

    def kmeans_cluster(self, k):
        """

        :param k:
        :return:
        """
        # load all data
        self.scr.prep_scr()

        kmeans = KMeans(n_clusters=k)
        print(self.scr.df_single_within_no_var)
        kmeans.fit(self.scr.df_single_within_no_var)

    def test_function(self):
        """"""
        # Step 3: Apply K-means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)  # Assuming 3 clusters
        kmeans.fit(self.scr.df_single_within_no_var)
        clusters = kmeans.labels_

        # Add the cluster labels to the DataFrame
        self.scr.df_single_within_no_var['Cluster'] = clusters

        # Step 4: Visualize the Results
        # Reduce dimensions using PCA for visualization
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(self.scr.df_single_within_no_var)
        df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = clusters

        # Plot the clusters
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis', s=100, alpha=0.7)
        plt.title('K-means Clustering with 2D PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.show()


if __name__ == '__main__':
    kw = KWithin()
    #kw.kmeans_cluster(3)
    kw.test_function()
