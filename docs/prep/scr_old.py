"""
scr.py
Author: Luísa Hörlle de Castro
"""

# import modules
import pandas as pd


class SCR:
    def __init__(self):
        self.data = None
        self.df = None
        self.df_single = None
        self.data_mean = None

    # load data
    data = pd.read_csv("/Users/luisahorlledecastro/UNI/Bachelorarbeit/"
                       "BachelorarbeitProject/data/SCR_Single_corr_prep_all.csv", index_col=0)

    # data without mean column
    df = data.drop("mean_all", axis=1)

    # data not doubled
    df_single = df[df['var1'] <= df['var2']]

    # only var1, var2 and mean_all
    selected_columns = ['var1', 'var2', 'mean_all']
    data_mean = data[selected_columns]

# TODO update paths
