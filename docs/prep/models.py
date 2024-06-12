"""
models.py
Author: Luísa Hörlle de Castro
"""

# import modules
import pandas as pd


class Models:
    def __init__(self):
        self.AK = None
        self.AKinv = None
        self.NN = None


# load ak model
Models.AK = pd.read_csv("/Users/luisahorlledecastro/UNI/Bachelorarbeit/"
                        "BachelorarbeitProject/data/Behav_Single_AK_corr_prep_all.csv")

# load ak inverse model
Models.AKinv = pd.read_csv("/Users/luisahorlledecastro/UNI/Bachelorarbeit/"
                           "BachelorarbeitProject/data/Behav_Single_invAK_corr_prep_all.csv")

# load nn model
Models.NN = pd.read_csv("/Users/luisahorlledecastro/UNI/Bachelorarbeit/"
                        "BachelorarbeitProject/data/Behav_Single_NN_corr_prep_all.csv")

# TODO update paths
