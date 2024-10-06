from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on the given dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method performs the analysis and visualizes the results.
        """ 
        pass
    
# Concrete strategy for numerical features
#-----------------------------------
#This strategy performs univariate analysis on numerical features.

class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on the given numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the numerical feature.
        feature (str): The name of the numerical feature to be analyzed.

        Returns:
        None: Displays a histogram with a KDE plot  .
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f'Histogram and KDE Plot for {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()
        
# Concrete strategy for categorical features
#-----------------------------------
#This strategy performs univariate analysis on categorical features.

class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on the given categorical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the categorical feature.
        feature (str): The name of the categorical feature to be analyzed.

        Returns:
        None: Displays a countplot for the categorical feature.
        """ 
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette='muted')
        plt.title(f'Countplot for {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()  

#Context class to perform univariate analysis
#-----------------------------------
#This class uses the strategy pattern to perform univariate analysis on a given feature.

class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initialize the UnivariateAnalyzer with a specific strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.
        
        Returns:
        None: Initializes the analyzer with a specific strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Set the strategy for univariate analysis.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None: Sets the strategy for univariate analysis.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Execute the analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        feature (str): The name of the feature to be analyzed.

        Returns:
        None: Performs the analysis and displays the results.
        """
        
        self._strategy.analyze(df, feature)
        
        
#Example usage
#-------------
#This example demonstrates how to use the UnivariateAnalyzer to perform univariate analysis on a numerical feature.

if __name__ == "__main__":
    # # Read the dataset
    # df = pd.read_csv(r"C:\Users\rohit\VSCode\End_to_End_Production_Project\prices-predictor-system\extracted_data\AmesHousing.csv")

    # #Analyze the numerical feature 'SalePrice'
    # analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    # analyzer.execute_analysis(df, 'SalePrice')

    # #Analyze the categorical feature 'Neighborhood'
    # analyzer.set_strategy(CategoricalUnivariateAnalysis())
    # analyzer.execute_analysis(df, 'Neighborhood')
    
    pass

