"""
Data prep psychopathological data
Author: Luísa Hörlle de Castro
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_p = pd.read_csv("../data/scr_subdef.csv")

print(max(data_p["Q_BDI_sum"]))
print(min(data_p["Q_BDI_sum"]))

