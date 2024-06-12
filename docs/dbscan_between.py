import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import itertools
from copy import deepcopy

# import modules
from prep.scr import SCR

class DBSCANCluster:
    def __init__(self):
        self.final_df = pd.DataFrame()
        self.scr_with_cluster = None
        self.cluster_means = pd.DataFrame()
        self.ch_index = None
        self.db_index = None
        self.silhouette = None
        self.eps = None
        self.min_samples = None
        self.cluster_dfs = None
        self.best_params = {'eps': None, 'min_samples': None}
        self.best_s = {'score': -1, 'eps': 0, 'min_samples': 0}
        self.best_d = {'score': float('inf'), 'eps': 0, 'min_samples': 0}
        self.best_c = {'score': -1, 'eps': 0, 'min_samples': 0}

    def plot_k_distance_graph(self, data, min_samples):
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(data)
        distances, indices = neighbors_fit.kneighbors(data)
        distances = np.sort(distances[:, -1], axis=0)
        plt.plot(distances)
        plt.title('K-Distance Graph')
        plt.xlabel('Data Points sorted by distance to {}th nearest neighbor'.format(min_samples))
        plt.ylabel('{}th Nearest Neighbor Distance'.format(min_samples))
        plt.show()

    def dbscan_cluster(self, data, eps, min_samples):
        data.columns = data.columns.astype(str)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data)
        data['cluster'] = dbscan.labels_
        self.cluster_dfs = []
        for cluster_label in set(dbscan.labels_):
            if cluster_label != -1:
                cluster_df = data[data['cluster'] == cluster_label].drop('cluster', axis=1)
                self.cluster_dfs.append(cluster_df)
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_metrics(data, dbscan, eps, min_samples)

    def dbscan_metrics(self, data, model, eps, min_samples):
        labels = model.labels_
        if len(set(labels)) > 1 and -1 not in labels:
            self.silhouette = silhouette_score(data, labels)
            self.db_index = davies_bouldin_score(data, labels)
            self.ch_index = calinski_harabasz_score(data, labels)
        else:
            self.silhouette = -1
            self.db_index = float('inf')
            self.ch_index = -1

    def dbscan_evaluation(self, data, eps_range, min_samples_values):
        data.columns = data.columns.astype(str)
        for eps, min_samples in itertools.product(eps_range, min_samples_values):
            self.dbscan_cluster(data, eps, min_samples)
            if self.silhouette > self.best_s['score']:
                self.best_s = {'score': self.silhouette, 'eps': eps, 'min_samples': min_samples}
            if self.db_index < self.best_d['score']:
                self.best_d = {'score': self.db_index, 'eps': eps, 'min_samples': min_samples}
            if self.ch_index > self.best_c['score']:
                self.best_c = {'score': self.ch_index, 'eps': eps, 'min_samples': min_samples}
        print(f"Best Silhouette Score: {self.best_s['score']} at eps = {self.best_s['eps']}, min_samples = {self.best_s['min_samples']}")
        print(f"Best Davies-Bouldin Index Score: {self.best_d['score']} at eps = {self.best_d['eps']}, min_samples = {self.best_d['min_samples']}")
        print(f"Best Calinski-Harabasz Score: {self.best_c['score']} at eps = {self.best_c['eps']}, min_samples = {self.best_c['min_samples']}")

    def show_cluster_heatmaps(self, data, eps=None, min_samples=None, target=None):
        self.scr_with_cluster = deepcopy(SCR.df)
        if eps is None or min_samples is None:
            eps = self.best_s['eps']
            min_samples = self.best_s['min_samples']
        self.dbscan_cluster(data, eps, min_samples)
        if target != "cluster":
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
                cluster_cols = cluster.T.columns
                cluster_data = self.scr_with_cluster[cluster_cols]
                cluster_mean = cluster_data.mean(axis=1)
                self.scr_with_cluster[f'Cluster {i + 1} Mean'] = cluster_mean
            for column in self.scr_with_cluster.columns[-len(self.cluster_dfs):]:
                heatmap_data = self.scr_with_cluster[self.scr_with_cluster['var1'] != self.scr_with_cluster['var2']]
                heatmap_data = self.scr_with_cluster.pivot_table(index='var1', columns='var2', values=column)
                sns.heatmap(heatmap_data, annot=False, cmap='coolwarm', cbar=True, vmin=2, vmax=2.6)
                plt.title(f'Heatmap of cluster {n}, column {column} with var1 and var2 as axes')
                plt.show()

    

    def complete_analysis(self, eps=None, min_samples=None):
        if eps is None or min_samples is None:
            eps = self.best_s['eps']
            min_samples = self.best_s['min_samples']
        self.show_cluster_heatmaps(SCR.df_single.drop(['var1', 'var2'], axis=1).T, eps, min_samples, target="cluster")
        self.anova_analysis(eps, min_samples)

if __name__ == '__main__':
    data = SCR.df_single.drop(['var1', 'var2'], axis=1).T
    dbscan_cluster = DBSCANCluster()
    dbscan_cluster.plot_k_distance_graph(data, min_samples=5)
    eps_values = np.arange(0.1, 5, 0.1)
    min_samples_values = range(3, 10)
    dbscan_cluster.dbscan_evaluation(data, eps_values, min_samples_values)
    dbscan_cluster.complete_analysis()
