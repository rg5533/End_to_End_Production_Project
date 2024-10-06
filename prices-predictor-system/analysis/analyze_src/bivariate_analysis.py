from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on the given dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        feature1 (str): The first feature to be analyzed.
        feature2 (str): The second feature to be analyzed.

        Returns:
        None: Displays the analysis results.
        """
        pass
    
#Concrete strategy for numerical vs numerical analysis
#------------------------------------------------
#This strategy is used to analyze the relationship between two numerical features using scatter plot.

class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    """
    Analyze the relationship between two numerical features using scatter plot.

    Parameters:
    df (pd.DataFrame): The dataframe to be analyzed.
    feature1 (str): The first numerical feature.
    feature2 (str): The second numerical feature.

    Returns:
    None: Displays the scatter plot.
    """
    
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f'Scatter Plot of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
        
        
#Concrete strategy for categorical vs numerical analysis
#------------------------------------------------
#This strategy is used to analyze the relationship between a categorical feature and a numerical feature using box plot.

class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Analyze the relationship between a categorical feature and a numerical feature using bar plot.
        Parameters: 
        df (pd.DataFrame): The dataframe to be analyzed.
        feature1 (str): The categorical feature.
        feature2 (str): The numerical feature.

        Returns:
        None: Displays the box plot.
        """
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f'Box Plot of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()
    
#Context class to perform bivariate analysis
#-----------------------------------
#THis class allows to switch between different bivariate analysis strategies.

class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initialize the BivariateAnalyzer with a specific strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Set the strategy to be used for bivariate analysis.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.
        """
        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on the given dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        feature1 (str): The first feature to be analyzed.
        feature2 (str): The second feature to be analyzed.

        Returns:
        None: Displays the analysis results.
        """
        self._strategy.analyze(df, feature1, feature2)  
        

#Example usage

if __name__ == "__main__":
    #Load the dataframe
    pass