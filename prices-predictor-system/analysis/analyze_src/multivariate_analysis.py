from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MultivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame):
        """
        Perform multivariate analysis on the given dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: orchestrates multivariate analysis process.
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
        
    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate a correlation heatmap for the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Displays the correlation heatmap.
        """
        pass
        
    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate a pairplot for the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Displays the pairplot.
        """
        pass

# Concrete Class for Multivariate Analysis with Correlation Heatmap and Pair Plot
# -------------------------------------------------------------------------------
# This class implements the methods to generate a correlation heatmap and a pair plot.
class SimpleMultivariateAnalysis(MultivariateAnalysisStrategy):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates and displays a correlation heatmap for the numerical features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a heatmap showing correlations between numerical features.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generates and displays a pair plot for the selected features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a pair plot for the selected features.
        """
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example usage of the SimpleMultivariateAnalysis class.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Perform Multivariate Analysis
    # multivariate_analyzer = SimpleMultivariateAnalysis()

    # Select important features for pair plot
    # selected_features = df[['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built']]

    # Execute the analysis
    # multivariate_analyzer.analyze(selected_features)
    pass
        

