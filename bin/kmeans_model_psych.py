"""
kmeans_model_psych.py
"""

# import libraries
from sklearn.cluster import KMeans
import pandas as pd
import logging

# import modules
from prep.prep_psychological import PrepPsychological
from elbow_method import DetermineKClusters, PB

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load data
PP = PrepPsychological()
DATA = PP.scaled_data

# determine n
DKC = DetermineKClusters(DATA)
N = DKC.n


class KMeansModelPsych:
    """
    A class to perform KMeans clustering on psychological data and analyze the resulting clusters.

    Attributes:
        data (pd.DataFrame): The scaled psychological data for clustering.
        n_clusters (int): The number of clusters to form.
        scaled_data_with_clusters (pd.DataFrame): DataFrame with scaled data and cluster labels.
        data_with_clusters (pd.DataFrame): Original data with cluster labels added.
        cluster_centers (pd.DataFrame): DataFrame containing the centers of the clusters.
        cluster_vp_dict (dict): Dictionary mapping cluster labels to participant IDs.
        list_of_cluster_dfs (list): List of DataFrames for each cluster.
    """

    def __init__(self, data, n_clusters):
        """
        Initializes the KMeansModelPsych class by setting up the data and performing clustering.

        Args:
            data (pd.DataFrame): The scaled psychological data for clustering.
            n_clusters (int): The number of clusters to form.
        """
        self.data = data
        self.n_clusters = n_clusters
        self.scaled_data_with_clusters = self.perform_kmeans_clustering()
        self.data_with_clusters = self.add_clusters()
        self.cluster_centers = None
        self.cluster_vp_dict = self.create_dict()
        self.list_of_cluster_dfs = []

    def perform_kmeans_clustering(self):
        """
        Performs KMeans clustering on the provided psychological data.

        Returns:
            pd.DataFrame: A DataFrame with cluster labels added to the original data.
        """
        # initialize kmeans with the specified number of clusters
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)

        # copy data
        df = self.data.copy()

        # fit the model to the data
        kmeans.fit(df)

        # add the cluster labels to the original data
        clustered_data = df.copy()
        clustered_data['Cluster'] = kmeans.labels_

        # create a DataFrame for the cluster centers
        self.cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns)

        logging.info(f"KMeans clustering done.")

        return clustered_data

    def create_dict(self):
        """
        Creates a dictionary mapping cluster labels to participant IDs.

        Returns:
            dict: A dictionary with cluster labels as keys and lists of participant IDs as values.
        """
        dictionary = {}

        for i in range(self.n_clusters):
            dictionary[i] = self.scaled_data_with_clusters[self.scaled_data_with_clusters["Cluster"] == i].index.tolist()

        return dictionary

    def add_clusters(self):
        """
        Adds the cluster labels to the original unscaled data.

        Returns:
            pd.DataFrame: A DataFrame with the original data and cluster labels.
        """
        original_data = PP.data.copy()
        cluster_df = self.scaled_data_with_clusters["Cluster"]
        merged_df = original_data.join(cluster_df)

        return merged_df

    def get_cluster_statistics(self):
        """
        Computes a full statistical summary (mean, std, min, max, etc.) for each column grouped by cluster.

        Returns:
            pd.DataFrame: A DataFrame with a full statistical summary for each cluster.
        """
        df = self.data_with_clusters
        df.reset_index(inplace=True, drop=True)
        df.set_index('Cluster', inplace=True)

        compact_labels = {
            'Q_BDI_sum': 'BDI Total',
            'Q_STAIT_sum': 'STAI-T Total',
            'Q_CTQ_emotVernachl': 'CTQ Emot. Neglect',
            'Q_CTQ_emotMissbr': 'CTQ Emot. Abuse',
            'Q_CTQ_koerperlVernachl': 'CTQ Phys. Neglect',
            'Q_CTQ_koerperlMissh': 'CTQ Phys. Abuse',
            'Q_CTQ_sexMissbr': 'CTQ Sex. Abuse',
            'Q_CTQ_di': 'CTQ Dissociation',
            'Q_CTQ_sum': 'CTQ Total',
            'Q_LTE_di': 'LTE Dissociation',
            'Q_LTE_sum': 'LTE Total'
        }

        df.rename(columns=compact_labels, inplace=True)

        # group by 'Cluster' and apply describe
        grouped_describe = df.groupby(level='Cluster').apply(lambda x: x.describe())

        # move the first index to become the second index for columns using unstack
        reshaped_describe = grouped_describe.unstack(level=0)

        reshaped_describe = reshaped_describe.drop(["25%", "75%"], axis=0)

        return reshaped_describe.T


KMMP = KMeansModelPsych(DATA, N)

if __name__ == "__main__":
    print(KMMP.get_cluster_statistics())
