"""
elbow_method.py
"""

# import libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import logging

# import modules
from prep.prep_physiological import PrepPhysiologicalBetween

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load data
PB = PrepPhysiologicalBetween()  # prep data
DATA = PB.kmeans_data_single


class DetermineKClusters:
    """
    Class to determine the optimal number of clusters using the elbow method and silhouette analysis.

    Attributes:
        data (pd.DataFrame): The data to be used for clustering.
        n (int): The optimal number of clusters based on the elbow method.
    """

    def __init__(self, data):
        """
        Initializes the DetermineKClusters class.

        Args:
            data (pd.DataFrame): The input data for clustering.
        """
        self.data = data
        self.n = self.elbow_method(data)

    @staticmethod
    def elbow_method(df, max_k=10, plot=False):
        """
        Determines the optimal number of clusters using the elbow method.

        Args:
            df (pd.DataFrame): The input data for clustering.
            max_k (int): The maximum number of clusters to test (default is 10).
            plot (bool): Whether to show a plot for the elbow method.

        Returns:
            int: The optimal number of clusters.
        """
        inertia = []
        k_list = list(range(1, max_k + 1))

        for k in k_list:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(df)
            inertia.append(kmeans.inertia_)

        if plot:
            # plot the inertia against the number of clusters
            plt.figure(figsize=(10, 6))
            plt.plot(k_list, inertia, marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
            plt.xticks(k_list)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xlabel('Number of clusters', fontsize=12, fontweight='bold')
            plt.ylabel('Inertia', fontsize=12, fontweight='bold')
            plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
            plt.savefig('elbow_method_plot.png', dpi=300, bbox_inches='tight')
            plt.show()

        # determine the optimal number of clusters (elbow point)
        elbow_point = 1
        for i in range(1, len(inertia) - 1):
            if (inertia[i - 1] - inertia[i]) > (inertia[i] - inertia[i + 1]):
                elbow_point = i + 1
                break

        logging.info(f"Elbow method done. Optimal number of clusters is {elbow_point}")

        return elbow_point

    @staticmethod
    def silhouette_analysis(df, max_k=10, plot=False):
        """
        Performs silhouette analysis to determine the optimal number of clusters.

        Args:
            df (pd.DataFrame): The input data for clustering.
            max_k (int): The maximum number of clusters to test (default is 10).
            plot (bool): Whether to show the silhouette score plot.

        Returns:
            int: The number of clusters with the highest silhouette score.
        """
        silhouette_scores = []
        k_list = list(range(2, max_k + 1))  # silhouette score requires at least 2 clusters

        for k in k_list:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(df)
            score = silhouette_score(df, kmeans.labels_)
            silhouette_scores.append(score)

        if plot:
            # plot the silhouette scores against the number of clusters
            plt.figure(figsize=(10, 6))
            plt.plot(k_list, silhouette_scores, marker='o', color='red', linestyle='-', linewidth=2, markersize=8)
            plt.xticks(k_list)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xlabel('Number of clusters', fontsize=12, fontweight='bold')
            plt.ylabel('Silhouette Score', fontsize=12, fontweight='bold')
            plt.title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
            plt.savefig('silhouette_analysis_plot.png', dpi=300, bbox_inches='tight')
            plt.show()

        # determine the optimal number of clusters (maximum silhouette score)
        optimal_k = k_list[silhouette_scores.index(max(silhouette_scores))]
        logging.info(f"Silhouette analysis done. Optimal number of clusters is {optimal_k}")

        return optimal_k

    @staticmethod
    def combined_elbow_silhouette(df, max_k=10):
        """
        Creates a figure with both the elbow method and silhouette analysis as subplots.

        Args:
            df (pd.DataFrame): The input data for clustering.
            max_k (int): The maximum number of clusters to test (default is 10).

        Returns:
            tuple: The optimal number of clusters based on the elbow method and silhouette score.
        """
        inertia = []
        silhouette_scores = []
        k_list = list(range(2, max_k + 1))

        for k in k_list:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(df)
            inertia.append(kmeans.inertia_)

            # calculate silhouette score
            score = silhouette_score(df, kmeans.labels_)
            silhouette_scores.append(score)

        # determine the optimal number of clusters (elbow point)
        elbow_point = 1
        for i in range(1, len(inertia) - 1):
            if (inertia[i - 1] - inertia[i]) > (inertia[i] - inertia[i + 1]):
                elbow_point = i + 1
                break

        # determine the optimal number of clusters based on silhouette score
        optimal_k_silhouette = k_list[silhouette_scores.index(max(silhouette_scores))]

        # plotting both methods with labels and title
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

        # elbow method plot (A)
        ax1.plot(range(1, max_k + 1),
                 [KMeans(n_clusters=k, random_state=0).fit(df).inertia_ for k in range(1, max_k + 1)],
                 marker='o', color="mediumaquamarine", linestyle='-', linewidth=2, markersize=8)
        ax1.set_xticks(range(1, max_k + 1))
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.set_xlabel('Number of clusters', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Inertia', fontsize=12, fontweight='bold')
        ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold', loc='center')
        ax1.text(-0.05, 1.1, 'A', transform=ax1.transAxes, size=20, weight='bold')

        # silhouette analysis plot (B)
        ax2.plot(k_list, silhouette_scores, marker='o', color='lightcoral', linestyle='-', linewidth=2, markersize=8)
        ax2.set_xticks(k_list)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Number of clusters', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax2.set_title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold', loc='center')
        ax2.text(-0.05, 1.1, 'B', transform=ax2.transAxes, size=20, weight='bold')

        # adding the main title
        fig.suptitle('Methods for Determining the Optimal Number of Physiological Clusters', fontsize=16,
                     fontweight='bold')

        plt.savefig('combined_elbow_silhouette_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

        logging.info(f"Elbow method done. Optimal number of clusters is {elbow_point}")
        logging.info(f"Silhouette analysis done. Optimal number of clusters is {optimal_k_silhouette}")

        return elbow_point, optimal_k_silhouette


if __name__ == '__main__':
    DKC = DetermineKClusters(DATA)
    DKC.combined_elbow_silhouette(DATA)
