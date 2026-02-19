"""
prep_physiological.py
"""

# Import libraries
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import logging
from scipy.stats import shapiro

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# get the absolute path of the current script
base_dir = os.path.abspath(os.path.dirname(__file__))
# define the relative path from the script's directory to the data file
relative_path = os.path.join(base_dir, '..', '..', 'data', 'scr_data.csv')
logging.info(f"Constructed file path: {relative_path}")

# load data
DATA = pd.read_csv(relative_path)


class PrepPhysiologicalBetween:
    """
    Preprocess physiological data between subjects, handling scaling, duplicates, and transformations.

    Attributes:
        data (pd.DataFrame): Raw data with the 'Unnamed: 0' column dropped.
        scaled_data_double (pd.DataFrame): Scaled data with all columns.
        scaled_data_single (pd.DataFrame): Scaled data with duplicates removed.
        kmeans_data_double (pd.DataFrame): Scaled data without the 'mean_all' column.
        kmeans_data_single (pd.DataFrame): Scaled data without the 'mean_all' column and duplicates removed.
        mean_all_scaled_double (pd.DataFrame): Scaled data with 'var1', 'var2', and 'mean_all' columns.
        mean_all_scaled_single (pd.DataFrame): Scaled data with duplicates removed including 'var1', 'var2', and 'mean_all'.
    """

    def __init__(self):
        """
        Initializes PrepPhysiologicalBetween class, loads data, scales it, and removes duplicates.
        """
        self.data = DATA.drop(columns=['Unnamed: 0'])

        # scaled data initialization
        self.scaled_data_double = self.data
        self.scaled_data_single = self.remove_duplicates(self.data)

        # kmeans-compatible data without 'mean_all'
        self.kmeans_data_double = self.transform_data(self.scaled_data_double.drop(columns="mean_all"))
        self.kmeans_data_single = self.transform_data(self.scaled_data_single.drop(columns="mean_all"))

        # scaled data for 'var1', 'var2', and 'mean_all'
        self.mean_all_scaled_double = self.scaled_data_double[["var1", "var2", "mean_all"]]
        self.mean_all_scaled_single = self.remove_duplicates(self.mean_all_scaled_double)
        logging.info("Prepped all files.")

    @staticmethod
    def remove_duplicates(df):
        """
        Removes duplicate rows where 'var1' is less than or equal to 'var2'.

        Args:
            df (pd.DataFrame): DataFrame from which to remove duplicates.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        return df[df['var1'] <= df['var2']]

    @staticmethod
    def scale_data(df):
        """
        Scales specified columns using MinMaxScaler, leaving 'var1' and 'var2' unchanged.

        Args:
            df (pd.DataFrame): DataFrame to scale.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        # separate columns to scale and those to keep unchanged
        columns_to_scale = df.drop(columns=['var1', 'var2'])
        columns_not_to_scale = df[['var1', 'var2']]

        # scale the specified columns
        scaler = MinMaxScaler()
        scaled_columns = scaler.fit_transform(columns_to_scale)

        # convert scaled columns to DataFrame
        scaled_columns_df = pd.DataFrame(scaled_columns, columns=columns_to_scale.columns,
                                         index=columns_to_scale.index)

        # combine scaled and unscaled columns
        final_data = pd.concat([columns_not_to_scale, scaled_columns_df], axis=1)

        return final_data

    @staticmethod
    def transform_data(df):
        """
        Transforms data to set 'var1' and 'var2' as a multi-index.

        Args:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        # set 'var1' and 'var2' as multi-index
        df = df.set_index(['var1', 'var2'])

        return df.T

    def check_normality(self, alpha=0.05):
        """
        Checks for normality of columns using the Shapiro-Wilk test.

        Args:
            alpha (float, optional): Significance level for normality test. Default is 0.05.

        Prints:
            Columns that are not normally distributed.
        """
        df = self.scaled_data_double.T
        normal_columns = []

        # loop through numeric columns
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:  # test only numerical columns
                # perform Shapiro-Wilk test
                stat, p_value = shapiro(df[column])

                # check if p-value is greater than alpha
                if p_value > alpha:
                    normal_columns.append(column)

        # print columns not normally distributed
        if normal_columns:
            print("Columns that are not normally distributed:")
            for col in normal_columns:
                print(f"- {col}")
        else:
            print("No columns are normally distributed.")


if __name__ == "__main__":
    pb = PrepPhysiologicalBetween()
    pb.check_normality()
