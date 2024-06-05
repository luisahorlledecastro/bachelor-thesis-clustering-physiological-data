```python
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.cluster import KMeans


# import modules
from prep.scr_old import SCR
from prep.models import Models
from kmeans_old import KM
from gmm_between import GMMCluster
```

# Load data and prepare for analysis


```python
# load data
data = SCR.df_single.drop(['var1', 'var2'], axis=1).T

# show data
data.head()
```

# K-Means Analysis


```python
# initiate K-Means object from class KM
km = KM()
```


```python
# apply elbow method to estimate number of clusters
#km.elbow_method(data, 10)
```


```python
# apply K-Means algorithm to plausible values
km.k_evaluation(data, (2,10))
```


```python
# apply K-Means with 2 clusters
km.kmeans_cluster(data, 2)
```


```python
km.show_cluster_heatmaps(data, k=2, target="cluster")
```


```python
km.show_mean_heatmap(data)
```

# Gaussian Mixed Model Analysis


```python

```
