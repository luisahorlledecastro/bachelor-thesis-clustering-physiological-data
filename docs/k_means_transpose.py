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



# import modules
from prep.scr import SCR


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
        Takes data and applies tbe elbow method to determine the ideal number of clusters
        for k-means clustering.
        :param data: takes data in form SCR.df_single.drop(['var1', 'var2'], axis=1).T
        :return: graph with the elbow method
        """
        data.columns = data.columns.astype(str)

        # Find the ideal number of clusters using, for example, the elbow method
        # Here we loop through a range of k values and calculate the sum of squared distances for each k
        # Then we plot the results and visually inspect for an "elbow" point
        sum_of_squared_distances = []
        K = range(1, k_max + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans = kmeans.fit(data)
            sum_of_squared_distances.append(kmeans.inertia_)

        # Plot the elbow curve
        import matplotlib.pyplot as plt
        plt.plot(K, sum_of_squared_distances, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def kmeans_cluster(self, data, k):
        """

        :param data:
        :param k:
        :return:
        """
        # TODO add explanation on how to access clusters
        data.columns = data.columns.astype(str)

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        # Add cluster labels to the original DataFrame
        data['cluster'] = kmeans.labels_

        # Now you have cluster labels in the DataFrame, y
        # ou can separate the data into separate DataFrames for each cluster
        self.cluster_dfs = []
        for cluster_label in range(k):
            self.cluster_df = data[data['cluster'] == cluster_label].drop('cluster', axis=1)
            self.cluster_dfs.append(self.cluster_df)

        self.num_clusters = k

        self.k_metrics(data, kmeans, k)

        return None

    def k_metrics(self, data, model, k):
        """

        :return:
        """
        # Calculate clustering metrics
        self.silhouette = silhouette_score(data, model.labels_)
        self.db_index = davies_bouldin_score(data, model.labels_)
        self.ch_index = calinski_harabasz_score(data, model.labels_)

        # Print the metric scores
        #print(f"Silhouette Score {k}: {self.silhouette:.2f}")
        #print(f"Davies-Bouldin Index {k}: {self.db_index:.2f}")
        #print(f"Calinski-Harabasz Index {k}: {self.ch_index:.2f}")

    def k_evaluation(self, data, krange):
        """

        :param data:
        :param krange: tuple
        :return:
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

    def show_cluster_heatmaps(self, data, k=None, target=None):
        """

        :param data:
        :param k:
        :param target:
        :return:
        """

        self.scr_with_cluster = deepcopy(scr.df_within)

        if k is None:
            k = self.best_s['k']

        self.kmeans_cluster(data, k)

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
            """
            for cluster in self.cluster_dfs:
                for column in cluster.T.columns:

                n += 1
                vector = cluster.mean()
                print(len(vector))
                heatmap_data = SCR.df.pivot_table(index='var1', columns='var2', values=vector.reshape, aggfunc='mean')
                plt.figure(figsize=(4, 3))
                sns.heatmap(heatmap_data, annot=False, cmap='coolwarm', cbar=True)
                plt.title(f'Heatmap of cluster {n} with var1 and var2 as axes')
                plt.show()
            """
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

                heatmap_data = self.scr_with_cluster.pivot_table(index='var1', columns='var2', values=column)

                for i in range(len(heatmap_data.index)):
                    heatmap_data.iat[i, i] = float('nan')

                sns.heatmap(heatmap_data, annot=False, cbar=True, vmin=2, vmax=2.54)
                plt.title(f'Heatmap of cluster {n}, column {column} with var1 and var2 as axes')
                plt.show()

    def show_mean_heatmap(self, data):

        heatmap_data_average = SCR.data_mean.pivot_table(index='var1', columns='var2', values='mean_all')

        for i in range(len(heatmap_data_average.index)):
            heatmap_data_average.iat[i, i] = float('nan')

        plt.figure(figsize=(4, 3))
        sns.heatmap(heatmap_data_average, annot=False, cbar=True, vmin=2, vmax=2.54)
        plt.title('Heatmap of Average Values with var1 and var2 as axes')
        plt.show()

    def anova_analysis(self, k=None):
        """

        :return:
        """
        # todo change cluster label to numeric

        self.final_df[["var1", "var2"]] = self.scr_with_cluster[["var1", "var2"]]
        self.final_df = pd.concat([self.final_df, self.scr_with_cluster[self.scr_with_cluster.columns[-k:]]], axis=1)


        # Reset index to ensure it exists for melting
        self.final_df.reset_index(inplace=True)

        # Create a long format DataFrame for ANOVA
        melted_data = pd.melt(self.final_df, id_vars=['var1', 'var2'], value_vars=self.final_df.columns[-k:],)
        melted_data.columns = ['var1', 'var2', 'cluster', 'value']
        melted_data['cluster'] = pd.Categorical(melted_data['cluster']).codes
        print(melted_data)

        # Perform the ANOVA
        model = ols('value ~ C(var1) * C(var2) * C(cluster)', data=melted_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)


    def complete_analysis(self, k=None):
        """

        :param k:
        :return:
        """


if __name__ == '__main__':
    scr = SCR()
    scr.prep_scr()
    data = scr.df_single_within_no_var
    km = KM()
    KM.elbow_method(KM, data, 40)
    #km.k_evaluation(data, (2, 20))
    km.kmeans_cluster(data, 2)
    km.show_cluster_heatmaps(data, k=2, target="cluster")
    km.show_mean_heatmap(data)
    #km.anova_analysis(2)
