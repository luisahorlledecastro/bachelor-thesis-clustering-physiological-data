"""
gmm_clustering.py
Author: Luísa Hörlle de Castro
"""

# import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from copy import deepcopy
import statsmodels.api as sm
from statsmodels.formula.api import ols

# import modules
from prep.scr_old import SCR


class GMMCluster:
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

    def bic_aic_method(self, data, k_max):
        """
        Takes data and applies BIC and AIC method to determine the ideal number of clusters
        for GMM clustering.
        :param data: takes data in form SCR.df_single.drop(['var1', 'var2'], axis=1).T
        :param k_max: maximum number of clusters to evaluate
        :return: graph with the BIC and AIC scores
        """
        data.columns = data.columns.astype(str)

        # Find the ideal number of clusters using BIC and AIC scores
        bic_scores = []
        aic_scores = []
        K = range(1, k_max + 1)
        for k in K:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(data)
            bic_scores.append(gmm.bic(data))
            aic_scores.append(gmm.aic(data))

        # Plot the BIC and AIC scores
        plt.figure(figsize=(10, 5))
        plt.plot(K, bic_scores, 'bx-', label='BIC')
        plt.plot(K, aic_scores, 'rx-', label='AIC')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Score')
        plt.title('BIC and AIC Scores For Optimal k')
        plt.legend()
        plt.show()

    def gmm_cluster(self, data, k):
        """
        Clusters the data using Gaussian Mixture Model.
        :param data: DataFrame with data to cluster
        :param k: number of clusters
        :return: None
        """
        data.columns = data.columns.astype(str)

        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(data)

        # Predict cluster labels
        labels = gmm.predict(data)
        data['cluster'] = labels

        # Separate the data into separate DataFrames for each cluster
        self.cluster_dfs = []
        for cluster_label in range(k):
            cluster_df = data[data['cluster'] == cluster_label].drop('cluster', axis=1)
            self.cluster_dfs.append(cluster_df)

        self.num_clusters = k

        self.gmm_metrics(data.drop('cluster', axis=1), gmm, labels)

        return None

    def save_clusters(self):
        """

        :return:
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
        filename = f"gmm_clusters_{num_clusters}.csv"

        # Save the DataFrame to CSV
        final_df.to_csv(filename)

        print(f"DataFrame saved to {filename}")

    def gmm_metrics(self, data, model, labels):
        """
        Calculates clustering metrics.
        :param data: DataFrame with data and cluster labels
        :param model: trained GMM model
        :param labels: predicted labels
        :return: None
        """
        # Calculate clustering metrics
        self.silhouette = silhouette_score(data, labels)
        self.db_index = davies_bouldin_score(data, labels)
        self.ch_index = calinski_harabasz_score(data, labels)

        # Print the metric scores
        # print(f"Silhouette Score {k}: {self.silhouette:.2f}")
        # print(f"Davies-Bouldin Index {k}: {self.db_index:.2f}")
        # print(f"Calinski-Harabasz Index {k}: {self.ch_index:.2f}")

    def k_evaluation(self, data, krange):
        """
        Evaluates the best number of clusters based on metrics.
        :param data: DataFrame with data to evaluate
        :param krange: tuple with range of k values to evaluate
        :return: None
        """
        data.columns = data.columns.astype(str)

        for i in range(krange[0], krange[1]):
            self.gmm_cluster(data.copy(), int(i))
            if self.silhouette > self.best_s['score']:
                self.best_s = {'score': self.silhouette, 'k': i}
            if self.db_index < self.best_d['score']:
                self.best_d = {'score': self.db_index, 'k': i}
            if self.ch_index > self.best_c['score']:
                self.best_c = {'score': self.ch_index, 'k': i}

        print(f"Best Silhouette Score: {self.best_s['score']} at k = {self.best_s['k']}")
        print(f"Best Davies-Bouldin Index Score: {self.best_d['score']} at k = {self.best_d['k']}")
        print(f"Best Calinski-Harabasz Score: {self.best_c['score']} at k = {self.best_c['k']}")

    def show_cluster_heatmaps(self, data, k=None, target=None):
        """
        Generates heatmaps for the clusters.
        :param data: DataFrame with data to cluster
        :param k: number of clusters
        :param target: target variable for heatmap
        :return: None
        """
        self.scr_with_cluster = deepcopy(SCR.df)

        if k is None:
            k = self.best_s['k']

        self.gmm_cluster(data.copy(), k)

        if target != "cluster":
            # Create a heatmap for each participant
            n = 0
            for cluster in self.cluster_dfs:
                n += 1
                for column in cluster.T.columns:
                    heatmap_data = self.scr_with_cluster[self.scr_with_cluster['var1'] != self.scr_with_cluster['var2']]
                    heatmap_data = self.scr_with_cluster.pivot_table(index='var1', columns='var2', values=column, aggfunc='mean')
                    plt.figure(figsize=(4, 3))
                    sns.heatmap(heatmap_data, annot=False, cmap='coolwarm', cbar=True)
                    plt.title(f'Heatmap of cluster {n}, column {column} with var1 and var2 as axes')
                    plt.show()

        else:
            n = 0
            for i, cluster in enumerate(self.cluster_dfs):
                # Get column names from the cluster dataframe
                cluster_cols = cluster.T.columns
                # Filter self.scr_with_cluster to include only columns present in the cluster
                cluster_data = self.scr_with_cluster[cluster_cols]
                # Calculate the mean of the cluster data
                cluster_mean = cluster_data.mean(axis=1)
                # Add a new column with the cluster mean
                self.scr_with_cluster[f'Cluster {i + 1} Mean'] = cluster_mean

            for column in self.scr_with_cluster.columns[-k:]:
                n += 1
                heatmap_data = self.scr_with_cluster.pivot_table(index='var1', columns='var2', values=column)

                for i in range(len(heatmap_data.index)):
                    heatmap_data.iat[i, i] = float('nan')

                sns.heatmap(heatmap_data, annot=False, cbar=True, vmin=2, vmax=2.54, cmap="YlGnBu")
                plt.title(f'Heatmap of cluster {n} with var1 and var2 as axes')
                plt.show()

    def show_clustermaps(self, data, k=None, target=None):
        """
        Generates clustermaps for the clusters.
        :param data: DataFrame with data to cluster
        :param k: number of clusters
        :param target: target variable for heatmap
        :return: None
        """
        self.scr_with_cluster = deepcopy(SCR.df)

        if k is None:
            k = self.best_s['k']

        self.gmm_cluster(data.copy(), k)

        if target != "cluster":
            # Create a clustermap for each participant
            n = 0
            for cluster in self.cluster_dfs:
                n += 1
                for column in cluster.T.columns:
                    heatmap_data = self.scr_with_cluster.pivot_table(index='var1', columns='var2', values=column,
                                                                     aggfunc='mean')
                    plt.figure(figsize=(4, 3))
                    sns.clustermap(heatmap_data, annot=False, cmap='coolwarm', cbar=True, vmin=2, vmax=2.54)
                    plt.title(f'Clustermap of cluster {n}, column {column} with var1 and var2 as axes')
                    plt.show()

        else:
            n = 0
            for i, cluster in enumerate(self.cluster_dfs):
                # Get column names from the cluster dataframe
                cluster_cols = cluster.T.columns
                # Filter self.scr_with_cluster to include only columns present in the cluster
                cluster_data = self.scr_with_cluster[cluster_cols]
                # Calculate the mean of the cluster data
                cluster_mean = cluster_data.mean(axis=1)
                # Add a new column with the cluster mean
                self.scr_with_cluster[f'Cluster {i + 1} Mean'] = cluster_mean

            for column in self.scr_with_cluster.columns[-k:]:
                n += 1
                heatmap_data = self.scr_with_cluster.pivot_table(index='var1', columns='var2', values=column)

                heatmap_data = heatmap_data.dropna()  # Remove rows with missing values

                sns.clustermap(heatmap_data, annot=False, cbar=True, cmap="YlGnBu")
                plt.title(f'Clustermap of cluster {n} with var1 and var2 as axes')
                plt.show()

    def show_mean_heatmap(self, data):
        """
        Generates a heatmap for the mean values of the clusters.
        :param data: DataFrame with data to cluster
        :return: None
        """
        heatmap_data_average = SCR.data_mean.pivot_table(index='var1', columns='var2', values='mean_all')

        for i in range(len(heatmap_data_average.index)):
            heatmap_data_average.iat[i, i] = float('nan')

        plt.figure(figsize=(4, 3))
        sns.heatmap(heatmap_data_average, annot=False, cbar=True, cmap="YlGnBu")
        plt.title('Heatmap of Average Values with var1 and var2 as axes', fontsize=9)
        plt.show()

    def anova_analysis(self, k=None):
        """
        Performs ANOVA analysis on the clusters.
        :param k: number of clusters
        :return: None
        """
        # Ensure cluster label is numeric
        self.final_df[["var1", "var2"]] = self.scr_with_cluster[["var1", "var2"]]
        self.final_df = pd.concat([self.final_df, self.scr_with_cluster[self.scr_with_cluster.columns[-k:]]], axis=1)

        # Reset index to ensure it exists for melting
        self.final_df.reset_index(inplace=True)

        # Create a long format DataFrame for ANOVA
        melted_data = pd.melt(self.final_df, id_vars=['var1', 'var2'], value_vars=self.final_df.columns[-k:])
        melted_data.columns = ['var1', 'var2', 'cluster', 'value']
        melted_data['cluster'] = pd.Categorical(melted_data['cluster']).codes
        print(melted_data)

        # Perform the ANOVA
        model = ols('value ~ C(var1) * C(var2) * C(cluster)', data=melted_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)


if __name__ == '__main__':
    data = SCR.df_single.drop(['var1', 'var2'], axis=1).T
    gmm_cluster = GMMCluster()
    #gmm_cluster.bic_aic_method(data, 40)
    #gmm_cluster.k_evaluation(data, (2, 10))
    gmm_cluster.gmm_cluster(data, 2)
    #gmm_cluster.save_clusters()
    gmm_cluster.show_clustermaps(data, 2, target="cluster")
    #gmm_cluster.show_cluster_heatmaps(data, k=2, target="cluster")
    #gmm_cluster.show_cluster_heatmaps(data, k=2, target="participants")


    #gmm_cluster.show_mean_heatmap(data)
    #gmm_cluster.anova_analysis(2)
