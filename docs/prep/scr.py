"""
scr.py
Author: Luísa Hörlle de Castro
"""

# import modules
import pandas as pd


class SCR:
    def __init__(self):
        # prep between
        self.data_between = None
        self.df_between = None
        self.df_single_between = None
        self.df_single_between_no_var = None
        self.data_mean_between = None

        # prep within
        self.data_within = None
        self.df_within = None
        self.df_single_within = None
        self.df_single_within_no_var = None
        self.data_mean_within = None

    def prep_within(self):
        """

        :return:
        """
        # between subjects
        # load data
        self.data_within = pd.read_csv("/Users/luisahorlledecastro/UNI/Bachelorarbeit/"
                           "BachelorarbeitProject/data/SCR_Single_corr_prep_all.csv", index_col=0)

        # data without mean column
        self.df_within = self.data_within.drop("mean_all", axis=1)

        # data not doubled
        self.df_single_within = self.df_within[self.df_within['var1'] <= self.df_within['var2']]

        self.df_single_within_no_var = self.df_single_within.drop(["var1", "var2"], axis=1)

        # only var1, var2 and mean_all
        selected_columns = ['var1', 'var2', 'mean_all']
        self.data_mean_within = self.data_within[selected_columns]

    def prep_between(self):
        """

        :return:
        """
        # full data with var1, var2, participants and mean all as rows
        self.data_between = self.data_within.T

        self.df_between = self.data_within.T

        self.df_single_between = self.df_single_within.T

        self.df_single_between_no_var = self.df_single_within_no_var.T

        self.data_mean_between = self.data_mean_within.T

    def prep_scr(self):
        """

        :return:
        """
        self.prep_within()
        self.prep_between()


# TODO update paths
if __name__ == '__main__':
    scr = SCR()
    scr.prep_scr()
