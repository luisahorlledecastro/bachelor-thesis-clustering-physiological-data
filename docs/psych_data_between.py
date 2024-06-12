"""
psych_data_between.py
Author: Luísa Hörlle de Castro
"""

# Import packages
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from copy import deepcopy
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules
from prep.scr_old import SCR

psych_data = pd.read_csv("../data/scr_subdef.csv").drop("Unnamed: 0", axis=1).dropna()

print(psych_data)



class KM:
    def __init__(self):
        self.final_df = pd.DataFrame()
        self.scr_with_cluster = None
        self.cluster_means = pd.DataFrame()
        self.ch_index = None
        self.db_index = None
        self.silhouette = None
        self.num_clusters = None
        self.cluster_dfs = None
        self.best_s = {'score': -1, 'k': 0}  # Initialize to a very low value
        self.best_d = {'score': float('inf'), 'k': 0}  # Initialize to a very high value
        self.best_c = {'score': -1, 'k': 0}  # Initialize to a very low value

    def elbow_method(self, data, k_max):
        """
        Takes data and applies the elbow method to determine the ideal number of clusters
        for k-means clustering.
        :param data: DataFrame for clustering
        :param k_max: Maximum number of clusters to test
        :return: graph with the elbow method
        """
        data.columns = data.columns.astype(str)

        sum_of_squared_distances = []
        K = range(1, k_max + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans = kmeans.fit(data)
            sum_of_squared_distances.append(kmeans.inertia_)

        plt.plot(K, sum_of_squared_distances, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def kmeans_cluster(self, data, k):
        """
        Performs k-means clustering on the data
        :param data: DataFrame for clustering
        :param k: Number of clusters
        """

        data.set_index('ID', inplace=True)
        data.columns = data.columns.astype(str)

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        # Add cluster labels to the original DataFrame
        data['cluster'] = kmeans.labels_

        # Separate the data into separate DataFrames for each cluster
        self.cluster_dfs = []
        for cluster_label in range(k):
            cluster_df = data[data['cluster'] == cluster_label].drop('cluster', axis=1)
            self.cluster_dfs.append(cluster_df)

        self.num_clusters = k

        self.k_metrics(data, kmeans, k)

    def save_clusters(self):
        """
        Save the cluster data to a CSV file
        """
        # Create a list to store DataFrames with double index
        dfs_with_double_index = []

        # Iterate through each cluster DataFrame
        for i, cluster_df in enumerate(self.cluster_dfs):
            # Assign cluster number as the second level of MultiIndex
            cluster_df.index = pd.MultiIndex.from_product([[f'Cluster {i + 1}'], cluster_df.index],
                                                          names=['Cluster', None])
            dfs_with_double_index.append(cluster_df)

        # Concatenate all DataFrames into a single DataFrame
        final_df = pd.concat(dfs_with_double_index)

        # Define the number of clusters
        num_clusters = len(self.cluster_dfs)

        # Define the filename
        filename = f"psych_clusters_{num_clusters}.csv"

        # Save the DataFrame to CSV
        final_df.to_csv(filename)

        print(f"DataFrame saved to {filename}")

    def k_metrics(self, data, model, k):
        """
        Calculate clustering metrics
        :param data: DataFrame for clustering
        :param model: KMeans model
        :param k: Number of clusters
        """
        self.silhouette = silhouette_score(data, model.labels_)
        self.db_index = davies_bouldin_score(data, model.labels_)
        self.ch_index = calinski_harabasz_score(data, model.labels_)

    def k_evaluation(self, data, krange):
        """
        Evaluate clustering for a range of k values
        :param data: DataFrame for clustering
        :param krange: Tuple of min and max k values to evaluate
        """
        data.columns = data.columns.astype(str)

        for i in range(krange[0], krange[1]):
            self.kmeans_cluster(data, int(i))
            if self.silhouette > self.best_s['score']:
                self.best_s = {'score': self.silhouette, 'k': i}
            if self.db_index < self.best_d['score']:
                self.best_d = {'score': self.db_index, 'k': i}
            if self.ch_index > self.best_c['score']:
                self.best_c = {'score': self.ch_index, 'k': i}

        print(f"Best Silhouette Score: {self.best_s['score']} at k = {self.best_s['k']}")
        print(f"Best Davies-Bouldin Index Score: {self.best_d['score']} at k = {self.best_d['k']}")
        print(f"Best Calinski-Harabasz Score: {self.best_c['score']} at k = {self.best_c['k']}")

    def anova_analysis(data, self, k=None):
        """
        Perform ANOVA analysis on the clustered data
        :param k: Number of clusters
        """
        # Create a copy of the data with cluster information
        self.scr_with_cluster = deepcopy(SCR.df)

        if k is None:
            k = self.best_s['k']

        self.kmeans_cluster(data, k)

        self.final_df[["var1", "var2"]] = self.scr_with_cluster[["var1", "var2"]]
        self.final_df = pd.concat([self.final_df, self.scr_with_cluster[self.scr_with_cluster.columns[-k:]]], axis=1)

        # Reset index to ensure it exists for melting
        self.final_df.reset_index(inplace=True)

        # Create a long format DataFrame for ANOVA
        melted_data = pd.melt(self.final_df, id_vars=['var1', 'var2'], value_vars=self.final_df.columns[-k:])
        melted_data.columns = ['var1', 'var2', 'cluster', 'value']
        melted_data['cluster'] = pd.Categorical(melted_data['cluster']).codes

        # Perform the ANOVA
        model = ols('value ~ C(var1) * C(var2) * C(cluster)', data=melted_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)

    def complete_analysis(data, self, k=None):
        """
        Perform a complete analysis
        :param k: Number of clusters
        """
        if k is None:
            self.elbow_method(data, 40)
            self.k_evaluation(data, (2, 20))
            k = self.best_s['k']
        self.kmeans_cluster(data, k)
        self.save_clusters()
        self.anova_analysis(k)

    def plot_clusters(self, data):
        """
        Visualize the clustered data
        :param data: DataFrame for clustering
        """
        data.columns = data.columns.astype(str)

        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(data)

        data['cluster'] = kmeans.labels_

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue='cluster', palette='viridis', data=data)
        plt.title('Cluster Visualization')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()


if __name__ == '__main__':
    km = KM()
    # KM.elbow_method(data, 40)
    # km.k_evaluation(data, (2, 20))
    km.kmeans_cluster(psych_data, 2)
    km.save_clusters()
    km.plot_clusters(psych_data)
    #km.anova_analysis(2)
