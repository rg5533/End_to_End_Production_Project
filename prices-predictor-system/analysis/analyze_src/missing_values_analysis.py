from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Abstract base class for missing values analysis strategy
#------------------------------------------------
#This class defines an abstract base class for missing values analysis strategies.
#Subclasses must implement the method to identify and visualize missing values in the dataset.

class MissingValuesAnalysisStrategy(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)
        
    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None
        """
        pass
    
    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.    

        Returns:
        None
        """
        pass
    
#Concrete Class for missing values analysis
#------------------------------------------
#This class implements methods to identify and visualize missing values in the dataframe.

class SimpleMissingValuesAnalysis(MissingValuesAnalysisStrategy):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None
        """
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum()
        print("Missing values:")
        print(missing_values[missing_values > 0])
        
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Displays a heatmap of missing values.
        """
        print("\nVisualizing Missing Values:")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()
        
    
#Example usage

if __name__ == "__main__":
    # df = pd.read_csv(r"C:\Users\rohit\VSCode\End_to_End_Production_Project\prices-predictor-system\extracted_data\AmesHousing.csv")
    # strategy = SimpleMissingValuesAnalysis()
    # strategy.analyze(df)
    pass

