"""
compare_clusters.py
"""

# import libraries
import logging
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# import modules
from kmeans_model import KMM
from kmeans_model_psych import KMMP

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CompareClusters:
    """
    A class to compare physiological and psychological clusters using statistical tests and visualization methods.

    Attributes:
        physiological_data_double (pd.DataFrame): Physiological data with duplicates.
        psychological_data (pd.DataFrame): Psychological data for analysis.
        physiological_clusters (pd.DataFrame): Data with physiological cluster labels.
        psychological_clusters (pd.DataFrame): Data with psychological cluster labels.
        physiological_clusters_dict (dict): Dictionary mapping physiological clusters to participants.
        psychological_clusters_dict (dict): Dictionary mapping psychological clusters to participants.
        compare_clusters_df (pd.DataFrame): Combined DataFrame for comparing physiological and psychological clusters.
    """

    def __init__(self, KMM, KMMP):
        """
        Initializes the CompareClusters class by loading physiological and psychological data,
        cluster labels, and creating a combined DataFrame for comparison.

        Args:
            KMM (KMeansModel): Instance of the KMeansModel for physiological data.
            KMMP (KMeansModelPsych): Instance of the KMeansModelPsych for psychological data.
        """
        self.physiological_data_double = KMM.data_double
        self.psychological_data = KMMP.data
        self.physiological_clusters = KMM.data_with_clusters
        self.psychological_clusters = KMMP.data_with_clusters
        self.physiological_clusters_dict = KMM.cluster_vp_dict
        self.psychological_clusters_dict = KMMP.cluster_vp_dict
        self.physiological_double_cluster = None
        self.psychological_double_cluster = None
        self.compare_clusters_df = self.create_full_df()

    def create_full_df(self, psych_values=False):
        """
        Creates a combined DataFrame of physiological and psychological cluster labels for comparison.

        Args:
            psych_values (bool): Whether to include psychological values in the resulting DataFrame (default is False).

        Returns:
            pd.DataFrame: A DataFrame containing physiological and psychological clusters for each participant.
        """
        dict1 = self.physiological_clusters_dict
        dict2 = self.psychological_clusters_dict

        # flatten dictionaries and create value-key pairs
        flat_dict1 = {v: k for k, lst in dict1.items() for v in lst}
        flat_dict2 = {v: k for k, lst in dict2.items() for v in lst}

        # identify overlapping participants
        overlap_values = set(flat_dict1.keys()).intersection(set(flat_dict2.keys()))

        # create lists for corresponding clusters and participants
        keys1, keys2, overlap_list = [], [], []

        for value in overlap_values:
            keys1.append(flat_dict1[value])
            keys2.append(flat_dict2[value])
            overlap_list.append(value)

        # create DataFrame with physiological and psychological clusters
        df = pd.DataFrame({
            'phys_clusters': keys1,
            'psych_clusters': keys2,
            'VP': overlap_list
        })

        df_psych = self.psychological_clusters

        if psych_values:
            df_psych.index.name = "VP"
            df = pd.merge(df, df_psych, how='outer', on='VP').drop("Cluster", axis=1)

        logging.info("Created full comparison DataFrame.")
        return df

    def matching_clusters(self):
        """
        Performs a t-test to check for significant differences between physiological and psychological clusters,
        and calculates the percentage of matching clusters between the two datasets.
        """
        sample1 = self.compare_clusters_df['phys_clusters']
        sample2 = self.compare_clusters_df['psych_clusters']

        t_stat, p_value = stats.ttest_ind(sample1, sample2)

        logging.info(f"T-statistic: {t_stat}, P-value: {p_value}")

        alpha = 0.05  # significance level
        if p_value < alpha:
            logging.info("Reject the null hypothesis: Significant difference between the two samples.")
        else:
            logging.info("Fail to reject the null hypothesis: No significant difference between the two samples.")

        # calculate percentage of matching clusters
        matches = sample1 == sample2
        percentage_matches = np.mean(matches) * 100
        logging.info(f"Percentage of matching clusters: {percentage_matches:.2f}%")

    def cohen(self):
        """
        Calculates Cohen's d for comparing physiological and psychological clusters,
        along with t-test statistics.
        """
        sample1 = self.compare_clusters_df['phys_clusters']
        sample2 = self.compare_clusters_df['psych_clusters']

        t_stat, p_value = stats.ttest_ind(sample1, sample2)

        logging.info(f"T-statistic: {t_stat}, P-value: {p_value}")

        # calculate Cohen's d
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)

        # pooled standard deviation
        n1, n2 = len(sample1), len(sample2)
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

        cohen_d = (mean1 - mean2) / pooled_std

        logging.info(f"Cohen's d: {cohen_d}")

        alpha = 0.05  # significance level
        if p_value < alpha:
            logging.info("Reject the null hypothesis: Significant difference between the two samples.")
        else:
            logging.info("Fail to reject the null hypothesis: No significant difference.")

    def plot_cluster_densities(self, show_individual=True):
        """
        Plots density distributions for all columns, comparing clusters across features.
        Displays individual plots or all plots together based on input.

        Args:
            show_individual (bool): Whether to display each feature's plot individually (default is True).
        """
        df = self.create_full_df(psych_values=True)

        clusters = df["phys_clusters"].unique()

        columns = df.columns[3:]

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

        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
        colors = ["purple", "lightblue"] + sns.color_palette("husl", len(clusters))

        if show_individual:
            for column in columns:
                plt.figure(figsize=(10, 6))

                for i, cluster in enumerate(clusters):
                    cluster_data = df[df["phys_clusters"] == cluster]

                    sns.histplot(cluster_data[column], label=f'Cluster {cluster}',
                                fill=True, alpha=0.6, linewidth=2, color=colors[i])

                compact_label = compact_labels.get(column, column)
                plt.title(f'Density Distribution of {compact_label} Across Clusters', fontweight='bold')
                plt.xlabel(compact_label, fontweight='bold')
                plt.ylabel('Density', fontweight='bold')

                plt.legend(title='Cluster', fontsize=12, title_fontsize='14')
                plt.grid(True, linestyle='--', linewidth=0.5)

                ax = plt.gca()
                for spine in ax.spines.values():
                    spine.set_color('black')

                plt.tight_layout()
                plt.show()
                plt.close()
        else:
            n_cols = 2
            n_rows = (len(columns) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
            axes = axes.flatten()

            for idx, column in enumerate(columns):
                ax = axes[idx]

                for i, cluster in enumerate(clusters):
                    cluster_data = df[df["phys_clusters"] == cluster]

                    sns.histplot(cluster_data[column], label=f'Cluster {cluster}', fill=True, alpha=0.6,
                                linewidth=2, color=colors[i], ax=ax, kde=True)

                compact_label = compact_labels.get(column, column)
                ax.set_title(f'Density Distribution of {compact_label}', fontweight='bold')
                ax.set_xlabel(compact_label, fontweight='bold')
                ax.set_ylabel('Density', fontweight='bold')

                ax.legend(title='Cluster', fontsize=10, title_fontsize='12')
                ax.grid(True, linestyle='--', linewidth=0.5)

                for spine in ax.spines.values():
                    spine.set_color('black')

            if len(columns) < len(axes):
                for i in range(len(columns), len(axes)):
                    fig.delaxes(axes[i])

            plt.tight_layout()
            plt.show()

    def cohen_all(self):
        """
        Calculates Cohen's d for each feature comparing physiological and psychological clusters.
        """
        df = self.create_full_df(psych_values=True)

        for column in df.columns[3:]:
            sample1 = df[df['psych_clusters'] == 0][column].dropna()
            sample2 = df[df['psych_clusters'] == 1][column].dropna()

            t_stat, p_value = stats.ttest_ind(sample1, sample2)

            mean1, mean2 = np.mean(sample1), np.mean(sample2)
            std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)

            n1, n2 = len(sample1), len(sample2)
            pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

            cohen_d = (mean1 - mean2) / pooled_std

            logging.info(f"--- {column} ---")
            logging.info(f"T-statistic: {t_stat}, P-value: {p_value}, Cohen's d: {cohen_d}")

            alpha = 0.05
            if p_value < alpha:
                logging.info(f"Reject the null hypothesis: Significant difference for {column}.")
            else:
                logging.info(f"Fail to reject the null hypothesis: No significant difference for {column}.\n")

    def plot_comparison_cluster_densities(self, save_plots=True, output_dir="plots"):
        """
        Plots histograms comparing physiological and psychological clusters for each feature.
        Optionally saves the plots to an output directory.

        Args:
            save_plots (bool): Whether to save the generated plots (default is True).
            output_dir (str): Directory to save plots (default is "plots").
        """
        df = self.create_full_df(psych_values=True)

        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        psych_clusters = df["psych_clusters"].dropna().unique().astype(int)
        phys_clusters = df["phys_clusters"].dropna().unique().astype(int)

        columns = df.columns[3:]

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

        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

        psych_colors = sns.color_palette("BuPu", len(psych_clusters))
        phys_colors = sns.color_palette("YlGnBu", len(phys_clusters))

        mid_point = len(columns) // 2
        column_groups = [columns[:mid_point], columns[mid_point:]]

        figure_titles = ['Comparison of Psychopathological vs Physiological Clusters (Part 1)',
                         'Comparison of Psychopathological vs Physiological Clusters (Part 2)']

        for fig_num, column_group in enumerate(column_groups):
            fig, axes = plt.subplots(len(column_group), 2, figsize=(15, len(column_group) * 5))
            axes = axes.flatten()

            for idx, column in enumerate(column_group):
                compact_label = compact_labels.get(column, column)

                ax_psych = axes[idx * 2]
                for i, cluster in enumerate(psych_clusters):
                    cluster_data = df[df["psych_clusters"] == cluster].dropna(subset=[column])

                    ax_psych.hist(cluster_data[column], bins=20, alpha=0.5, label=f'Cluster {int(cluster)+1}',
                                  color=psych_colors[i], density=True)
                    sns.kdeplot(cluster_data[column], color=psych_colors[i], ax=ax_psych, lw=2)

                ax_psych.set_title(f'Psychopathological Clusters - {compact_label}', fontweight='bold')
                ax_psych.set_xlabel(compact_label, fontweight='bold')
                ax_psych.set_ylabel('Density', fontweight='bold')
                ax_psych.legend(title='Cluster', fontsize=10, title_fontsize='12')
                ax_psych.grid(True, linestyle='--', linewidth=0.5)

                ax_phys = axes[idx * 2 + 1]
                for i, cluster in enumerate(phys_clusters):
                    cluster_data = df[df["phys_clusters"] == cluster].dropna(subset=[column])

                    ax_phys.hist(cluster_data[column], bins=20, alpha=0.5, label=f'Cluster {int(cluster)+1}',
                                 color=phys_colors[i], density=True)
                    sns.kdeplot(cluster_data[column], color=phys_colors[i], ax=ax_phys, lw=2)

                ax_phys.set_title(f'Physiological Clusters - {compact_label}', fontweight='bold')
                ax_phys.set_xlabel(compact_label, fontweight='bold')
                ax_phys.set_ylabel('Density', fontweight='bold')
                ax_phys.legend(title='Cluster', fontsize=10, title_fontsize='12')
                ax_phys.grid(True, linestyle='--', linewidth=0.5)

            fig.suptitle(figure_titles[fig_num], fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            if save_plots:
                plot_filename = os.path.join(output_dir, f"{figure_titles[fig_num].replace(' ', '_').lower()}.png")
                fig.savefig(plot_filename, dpi=300)
                logging.info(f"Saved plot: {plot_filename}")

            plt.show()

    def get_phys_cluster_statistics(self):
        """
        Computes the full statistical summary (mean, std, min, max, etc.) for each column grouped by physiological cluster.

        Returns:
            pd.DataFrame: A DataFrame with the full statistical summary for each physiological cluster.
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

        grouped_describe = df.groupby(level='Cluster').apply(lambda x: x.describe())
        reshaped_describe = grouped_describe.unstack(level=0)
        reshaped_describe = reshaped_describe.drop(["25%", "75%"], axis=0)

        logging.info("Computed statistical summary for physiological clusters.")
        return reshaped_describe.T

    def permutation(self, num_permutations=1000, cluster='psych_clusters'):
        """
        Runs permutation tests to compare distributions between physiological and psychological clusters.

        Args:
            num_permutations (int): Number of permutations to perform (default is 1000).
            cluster (str): The cluster type to compare ('phys_clusters' or 'psych_clusters') (default is 'psych_clusters').
        """
        df = self.create_full_df(psych_values=True)

        for column in df.columns[3:]:
            sample1 = df[df[cluster] == 0][column].dropna()
            sample2 = df[df[cluster] == 1][column].dropna()

            t_stat, p_value = stats.ttest_ind(sample1, sample2)

            mean1, mean2 = np.mean(sample1), np.mean(sample2)
            std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)

            n1, n2 = len(sample1), len(sample2)
            pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

            cohen_d = (mean1 - mean2) / pooled_std

            print(f"--- {column} ---")
            print(f"T-statistic: {t_stat}, P-value (t-test): {p_value}, Cohen's d: {cohen_d}")

            combined_data = np.concatenate([sample1, sample2])
            observed_diff = np.abs(mean1 - mean2)

            perm_diffs = []
            for _ in range(num_permutations):
                np.random.shuffle(combined_data)
                perm_sample1 = combined_data[:n1]
                perm_sample2 = combined_data[n1:]

                perm_diff = np.abs(np.mean(perm_sample1) - np.mean(perm_sample2))
                perm_diffs.append(perm_diff)

            perm_diffs = np.array(perm_diffs)
            perm_p_value = np.mean(perm_diffs >= observed_diff)

            logging.info(f"P-value (permutation test): {perm_p_value}")
            print(f"P-value (permutation test): {perm_p_value}")

            alpha = 0.05
            if p_value < alpha:
                print(f"Reject the null hypothesis: Significant difference for {column}.")
            else:
                print(f"Fail to reject the null hypothesis: No significant difference for {column}.\n")

            if perm_p_value < alpha:
                print(f"Reject the null hypothesis (Permutation Test): Significant difference for {column}.")
            else:
                print(
                    f"Fail to reject the null hypothesis (Permutation Test): No significant difference for {column}.\n")


if __name__ == "__main__":
    CC = CompareClusters(KMM, KMMP)
    CC.plot_comparison_cluster_densities()
    CC.cohen_all()
    CC.permutation(num_permutations=1000)
