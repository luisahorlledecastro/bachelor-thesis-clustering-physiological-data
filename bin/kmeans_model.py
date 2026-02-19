"""
kmeans_model.py
"""

# import libraries
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy import stats
import numpy as np

# import modules
from elbow_method import DetermineKClusters, PB
from kmeans_model_psych import KMMP

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load data
DATA_SINGLE = PB.kmeans_data_single
DATA_DOUBLE = PB.kmeans_data_double

# determine n
DKC = DetermineKClusters(DATA_SINGLE)
N = DKC.n


class KMeansModel:
    """
    A class to perform KMeans clustering and analyze clusters with various heatmaps and statistical tests.

    Attributes:
        data (pd.DataFrame): DataFrame for clustering.
        data_double (pd.DataFrame): DataFrame with duplicate entries.
        n_clusters (int): The number of clusters to form.
        data_with_clusters (pd.DataFrame): DataFrame with cluster labels added.
        cluster_centers (pd.DataFrame): DataFrame containing the centers of the clusters.
        cluster0 (pd.DataFrame): DataFrame containing the data for cluster 0.
        cluster_vp_dict (dict): Dictionary mapping cluster labels to participant IDs.
        list_of_cluster_dfs (list): List of DataFrames for each cluster.
    """

    def __init__(self, data_single, data_double, n_clusters):
        """
        Initializes the KMeansModel class by setting up the data and performing clustering.

        Args:
            data_single (pd.DataFrame): DataFrame for clustering without duplicates.
            data_double (pd.DataFrame): DataFrame with duplicate entries.
            n_clusters (int): The number of clusters to form.
        """
        self.data = data_single
        self.data_double = data_double
        self.n_clusters = n_clusters
        self.data_with_clusters = self.perform_kmeans_clustering()
        self.cluster_centers = None
        self.cluster0 = self.data_with_clusters[self.data_with_clusters["Cluster"] == 0]
        self.cluster_vp_dict = self.create_dict()
        self.list_of_cluster_dfs = []

    def perform_kmeans_clustering(self):
        """
        Performs KMeans clustering on the provided DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with cluster labels and cluster centers.
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
            dictionary[i] = self.data_with_clusters[self.data_with_clusters["Cluster"] == i].index.tolist()

        return dictionary

    @staticmethod
    def prep_for_heatmap(df):
        """
        Prepares the data for heatmap generation by calculating mean clusters and melting the data.

        Args:
            df (pd.DataFrame): The DataFrame to prepare for heatmap.

        Returns:
            pd.DataFrame: The prepared DataFrame for heatmap plotting.
        """
        df_t = df.T
        df_t['mean_cluster'] = df_t.mean(axis=1)

        df_t = df_t[df_t.index.get_level_values('var1') != df_t.index.get_level_values('var2')]

        melted_data = pd.melt(df_t.reset_index(), id_vars=['var1', 'var2'], value_vars=['mean_cluster'])

        heatmap_data = melted_data.pivot_table(index='var1', columns='var2', values='value')

        return heatmap_data

    def heatmaps(self, save=False, display_together=False):
        """
        Generates heatmaps for each cluster with customized styling.
        If display_together is True, displays all clusters side by side.

        Args:
            save (bool): Whether to save the generated heatmaps (default is False).
            display_together (bool): Whether to display heatmaps side by side (default is False).
        """
        df = self.data_double.copy()

        if display_together:
            fig, axes = plt.subplots(1, self.n_clusters, figsize=(
                10 * self.n_clusters, 8))  # adjust figure size based on number of clusters
            fig.suptitle('Cluster Heatmaps', fontsize=16, fontweight='bold')  # joint title for the subplots
            plt.subplots_adjust(top=0.85, wspace=0.3)  # adjust top padding and space between subplots

        for i in range(self.n_clusters):
            cluster = df[df.index.isin(self.cluster_vp_dict[i])]
            self.list_of_cluster_dfs.append(cluster)

            if display_together:
                ax = axes[i]
            else:
                plt.figure(figsize=(10, 8))
                ax = sns.heatmap(self.prep_for_heatmap(cluster), cmap="YlGnBu", annot=False,
                                 cbar_kws={'label': 'Similarity Scale'})

            heatmap_data = self.prep_for_heatmap(cluster)
            sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, ax=ax, cbar_kws={'label': 'Similarity Scale'})

            # clear x-axis label and set y-axis label
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_ylabel('Sorted Arousal', fontsize=12, fontweight='bold')

            # customization
            ax.set_yticks([])
            if display_together:
                ax.set_title(f'Cluster {i + 1}', fontsize=14, fontweight='bold')
            else:
                plt.title(f'Heatmap for Cluster {i + 1}', fontsize=14, fontweight='bold')

            # add "Low" at the top and "High" at the bottom of the y-axis
            ax.text(-0.1, 1.02, 'Low', transform=ax.transAxes, ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(-0.1, -0.02, 'High', transform=ax.transAxes, ha='center', va='center', fontsize=12,
                    fontweight='bold')

            # add arrow on y-axis
            ax.annotate(
                '', xy=(-0.1, 0.1), xycoords='axes fraction',
                xytext=(-0.1, 0.9), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='black', lw=1)
            )

            # customize the colorbar
            colorbar = ax.collections[0].colorbar
            colorbar.set_label('Similarity', fontsize=12, fontweight='bold', labelpad=15)  # adjust label spacing
            colorbar.ax.text(0.5, -0.05, 'Dissimilar', ha='center', va='center', fontsize=12, fontweight='bold',
                             transform=colorbar.ax.transAxes)
            colorbar.ax.text(0.5, 1.05, 'Similar', ha='center', va='center', fontsize=12, fontweight='bold',
                             transform=colorbar.ax.transAxes)

            # add labels "A", "B", etc.
            if display_together:
                label = chr(65 + i)  # convert 0 -> 'A', 1 -> 'B', etc.
                ax.text(-0.15, 1.1, label, transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center',
                        va='center')

            if save and not display_together:
                plt.savefig(f'heatmap_cluster_{i + 1}.png', dpi=300, bbox_inches='tight')
                logging.info(f"Saved heatmap for cluster {i + 1}.")

            if not display_together:
                plt.show()

        if display_together:
            if save:
                plt.savefig('combined_heatmap_clusters.png', dpi=300, bbox_inches='tight')
                logging.info("Saved combined cluster heatmaps.")
            plt.show()

    def mean_heatmap(self, save=False):
        """
        Generates a heatmap for the mean data with custom styling.

        Args:
            save (bool): Whether to save the generated heatmap (default is False).
        """
        df = self.data_double.copy()

        # prepare data for the heatmap
        heatmap_data = self.prep_for_heatmap(df)

        # create the heatmap with custom styling
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, cbar_kws={'label': 'Value Scale'})

        # customize the plot
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Mean Heatmap for All Participants', fontsize=14, fontweight='bold')
        plt.xlabel('', fontsize=12, fontweight='bold')
        plt.ylabel('Sorted Arousal', fontsize=12, fontweight='bold')

        # add "Low" at the top and "High" at the bottom of the y-axis
        ax.text(-0.1, 1.02, 'Low', transform=ax.transAxes, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(-0.1, -0.02, 'High', transform=ax.transAxes, ha='center', va='center', fontsize=12, fontweight='bold')

        ax.annotate(
            '', xy=(-0.1, 0.1), xycoords='axes fraction',
            xytext=(-0.1, 0.9), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->", color='black', lw=1)
        )

        # customize the colorbar
        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Similarity', fontsize=12, fontweight='bold', labelpad=15)
        colorbar.ax.text(0.5, -0.05, 'Dissimilar', ha='center', va='center', fontsize=12, fontweight='bold',
                         transform=colorbar.ax.transAxes)
        colorbar.ax.text(0.5, 1.05, 'Similar', ha='center', va='center', fontsize=12, fontweight='bold',
                         transform=colorbar.ax.transAxes)

        if save:
            plt.savefig('heatmap_mean.png', dpi=300, bbox_inches='tight')
            logging.info(f"Saved mean image.")

        plt.show()

    def general_characteristics(self):
        """
        Prints general characteristics of each cluster including the number of participants and mean correlation.

        Returns:
            None
        """
        clusters = self.cluster_vp_dict

        for n in range(self.n_clusters):
            mean_corr = (self.data_with_clusters[self.data_with_clusters["Cluster"] == n]
                         .drop("Cluster", axis=1).mean().mean())
            print(f"Cluster {n} contains {len(clusters[n])} participants and has a mean correlation of {round(mean_corr, 2)}.")

    def phys_psych_heatmaps(self, save=False, display_together=False):
        """
        Generates physiological heatmaps for each psychological cluster.
        If display_together is True, displays all clusters side by side.

        Args:
            save (bool): Whether to save the generated heatmaps (default is False).
            display_together (bool): Whether to display heatmaps side by side (default is False).
        """
        df = self.data_double.copy()

        if display_together:
            fig, axes = plt.subplots(1, KMMP.n_clusters, figsize=(
                10 * self.n_clusters, 8))  # adjust figure size based on number of clusters
            fig.suptitle('Psychopathological Cluster Heatmaps', fontsize=16, fontweight='bold')  # joint title for the subplots
            plt.subplots_adjust(top=0.85, wspace=0.3)  # adjust top padding and space between subplots

        for i in range(KMMP.n_clusters):
            cluster = df[df.index.isin(KMMP.cluster_vp_dict[i])]
            self.list_of_cluster_dfs.append(cluster)

            if display_together:
                ax = axes[i]
            else:
                plt.figure(figsize=(10, 8))
                ax = sns.heatmap(self.prep_for_heatmap(cluster), cmap="YlGnBu", annot=False,
                                 cbar_kws={'label': 'Similarity Scale'})

            heatmap_data = self.prep_for_heatmap(cluster)
            sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, ax=ax, cbar_kws={'label': 'Similarity Scale'})

            # clear x-axis label and set y-axis label
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_ylabel('Sorted Arousal', fontsize=12, fontweight='bold')

            # customization
            ax.set_yticks([])
            if display_together:
                ax.set_title(f'Cluster {i + 1}', fontsize=14, fontweight='bold')
            else:
                plt.title(f'Physiological Heatmap for Psychological Cluster {i + 1}', fontsize=14, fontweight='bold')

            # add "Low" at the top and "High" at the bottom of the y-axis
            ax.text(-0.1, 1.02, 'Low', transform=ax.transAxes, ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(-0.1, -0.02, 'High', transform=ax.transAxes, ha='center', va='center', fontsize=12,
                    fontweight='bold')

            # add arrow on y-axis
            ax.annotate(
                '', xy=(-0.1, 0.1), xycoords='axes fraction',
                xytext=(-0.1, 0.9), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='black', lw=1)
            )

            # customize the colorbar
            colorbar = ax.collections[0].colorbar
            colorbar.set_label('Similarity', fontsize=12, fontweight='bold', labelpad=15)  # adjust label spacing
            colorbar.ax.text(0.5, -0.05, 'Dissimilar', ha='center', va='center', fontsize=12, fontweight='bold',
                             transform=colorbar.ax.transAxes)
            colorbar.ax.text(0.5, 1.05, 'Similar', ha='center', va='center', fontsize=12, fontweight='bold',
                             transform=colorbar.ax.transAxes)

            # add labels "A", "B", etc.
            if display_together:
                label = chr(65 + i)  # convert 0 -> 'A', 1 -> 'B', etc.
                ax.text(-0.15, 1.1, label, transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center',
                        va='center')

            if save and not display_together:
                plt.savefig(f'heatmap_psych_cluster_{i + 1}.png', dpi=300, bbox_inches='tight')
                logging.info(f"Saved heatmap for cluster {i + 1}.")

            if not display_together:
                plt.show()

        if display_together:
            if save:
                plt.savefig('combined_heatmap_psych_clusters.png', dpi=300, bbox_inches='tight')
                logging.info("Saved combined cluster heatmaps.")
            plt.show()

    def compare_phys_clusters_stats(self):
        """
        Compares the two clusters to determine if they are significantly different using a t-test and Cohen's d.

        Returns:
            None
        """
        df = self.data_with_clusters.copy().reset_index(drop=True)

        c1 = df[df["Cluster"] == 0].drop(columns=["Cluster"]).mean()
        c2 = df[df["Cluster"] == 1].drop(columns=["Cluster"]).mean()

        # perform t-test
        t_stat, p_value = stats.ttest_ind(c1, c2, axis=0)

        # calculate Cohen's d
        mean1, mean2 = c1.mean(), c2.mean()
        std1, std2 = c1.std(), c2.std()
        n1, n2 = len(c1), len(c2)

        # pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

        # Cohen's d
        cohen_d = (mean1 - mean2) / pooled_std

        # print the results
        print(f"T-statistic: {t_stat}")
        print(f"P-value: {p_value}")
        print(f"Cohen's d: {cohen_d}")

    def permutation_test(self, num_permutations=10000):
        """
        Performs a permutation test to compare two sets of data.

        Args:
            num_permutations (int): Number of permutations to perform (default is 10,000).

        Returns:
            None
        """
        df = self.data_with_clusters.copy().reset_index(drop=True)

        data1 = df[df["Cluster"] == 0].drop(columns=["Cluster"]).mean()
        data2 = df[df["Cluster"] == 1].drop(columns=["Cluster"]).mean()

        # observed difference in means
        observed_diff = np.mean(data1) - np.mean(data2)

        # concatenate the data
        combined_data = np.concatenate([data1, data2])

        # initialize a counter for extreme values
        extreme_count = 0

        # permute the data num_permutations times
        for _ in range(num_permutations):
            # shuffle the combined data
            np.random.shuffle(combined_data)

            # split the shuffled data into two groups
            permuted_data1 = combined_data[:len(data1)]
            permuted_data2 = combined_data[len(data1):]

            # calculate the difference in means for the permuted groups
            permuted_diff = np.mean(permuted_data1) - np.mean(permuted_data2)

            # count how many times the permuted difference is as extreme as the observed difference
            if abs(permuted_diff) >= abs(observed_diff):
                extreme_count += 1

        # calculate the p-value
        p_value = extreme_count / num_permutations

        print(f"Observed Difference: {observed_diff}")
        print(f"P-value: {p_value}")


KMM = KMeansModel(DATA_SINGLE, DATA_DOUBLE, N)

if __name__ == '__main__':
    KMM.heatmaps(display_together=True)
    KMM.phys_psych_heatmaps(display_together=True)
    KMM.compare_phys_clusters_stats()
    KMM.permutation_test()
