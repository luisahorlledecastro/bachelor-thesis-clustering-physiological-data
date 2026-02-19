"""
prep_psychological.py
"""

# Import libraries
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# get the absolute path of the current script
base_dir = os.path.abspath(os.path.dirname(__file__))
# define the relative path from the script's directory to the data file
relative_path = os.path.join(base_dir, '..', '..', 'data', 'psychopathological_data.csv')
logging.info(f"Constructed file path: {relative_path}")

# load data
DATA = pd.read_csv(relative_path)


class PrepPsychological:
    """
    A class used to preprocess psychological data for analysis.

    Attributes:
        data (pd.DataFrame): The raw data with the 'Unnamed: 0' column dropped.
        scaled_data (pd.DataFrame): The scaled data excluding emotional neglect and emotional abuse columns.
    """

    def __init__(self):
        """
        Initializes the PrepPsychological class by loading, adjusting labels, scaling data, and removing duplicates.
        """
        # data
        self.data = self.adjust_labels(DATA.drop(columns=['Unnamed: 0']).dropna())

        # scaled with mean as last column
        self.scaled_data = self.scale_data(self.data).drop(columns=["Q_CTQ_emotMissbr", "Q_CTQ_emotVernachl"])

        # log
        logging.info("Prepped all files.")

    @staticmethod
    def adjust_labels(data):
        """
        Adjusts the labels of the data by appending 'VP' to the 'ID' column, setting 'ID' as the index,
        and removing duplicate entries.

        Args:
            data (pd.DataFrame): The DataFrame containing the data to adjust.

        Returns:
            pd.DataFrame: The adjusted DataFrame with 'ID' as the index and duplicates removed.
        """
        data["ID"] = "VP" + data["ID"]

        data = data.set_index("ID")

        data = data.drop_duplicates()

        return data

    @staticmethod
    def scale_data(df, scaler=MinMaxScaler()):
        """
        Scales the numerical columns of the DataFrame using the specified scaler (MinMaxScaler by default).

        Args:
            df (pd.DataFrame): The DataFrame containing the data to scale.
            scaler (sklearn.preprocessing): The scaler instance to apply to the data (default is MinMaxScaler).

        Returns:
            pd.DataFrame: The scaled DataFrame.
        """
        # copy the DataFrame to avoid modifying the original data
        df_scaled = df.copy()

        # scale each column separately
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:  # scale only numerical columns
                scaled_values = scaler.fit_transform(df[[column]])
                df_scaled[column] = scaled_values

        return df_scaled

    def analysis_feature_extraction(self, save=False):
        """
        Generates a correlation matrix heatmap for the scaled data. Optionally saves the plot as a PNG image.

        Args:
            save (bool): Whether to save the heatmap plot as a PNG file (default is False).
        """
        df = self.scaled_data

        # compute the correlation matrix
        corr_matrix = df.corr()

        # define a dictionary with compact labels for xticks and yticks
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

        # create a list of compact labels for xticks and yticks
        compact_labels_for_plot = [compact_labels.get(col, col) for col in corr_matrix.columns]

        # plotting the heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(corr_matrix, annot=True, cmap='BuPu', cbar=True, square=True, fmt='.2f')

        # get the number of columns for positioning the ticks
        n_labels = len(compact_labels_for_plot)

        # set the xticks and yticks in the middle of each square
        ax.set_xticks([x + 0.5 for x in range(n_labels)])  # shift by 0.5 to center the ticks
        ax.set_xticklabels(compact_labels_for_plot, rotation=45, ha='right', fontsize=10)

        ax.set_yticks([y + 0.5 for y in range(n_labels)])  # shift by 0.5 to center the ticks
        ax.set_yticklabels(compact_labels_for_plot, rotation=0, ha='right', fontsize=10)

        # add title and adjust layout
        plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            plt.savefig('psych_feature extraction.png', dpi=300, bbox_inches='tight')

        # display the heatmap
        plt.show()


if __name__ == "__main__":
    pp = PrepPsychological()
    pp.analysis_feature_extraction()
