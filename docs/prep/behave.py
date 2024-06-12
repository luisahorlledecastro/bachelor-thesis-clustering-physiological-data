'''
Author: Luísa Hörlle de Castro
'''

# import modules
import pandas as pd


# load data
data = pd.read_csv('/Users/luisahorlledecastro/UNI/Bachelorarbeit/'
                   'BachelorarbeitProject/data/scr_subdef.csv')

q_bdi_scale = 21*3

data_scaled = pd.DataFrame()

data_scaled['Q_BDI_sum'] = data['Q_BDI_sum'] / q_bdi_scale

print(data.head())
print(data_scaled.head())

# TODO update paths
