from abc import ABC, abstractmethod

import pandas as pd

# Abstract Base Class for Data Inspection Strategies
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
# Subclasses must implement the inspect method.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass                

# Concrete Strategy for Basic Data Inspection
# --------------------------------------------------
# This strategy inspects the data types of each column and counts non-null values.

class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-Null Counts:")
        print(df.info())

# Concrete Strategy for Basic Data Inspection# Concrete Strategy for Summary Statistics Inspection
# -----------------------------------------------------
# This strategy provides summary statistics for both numerical and categorical features.
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=['O']))
        

# Context Class that uses a DataInspectionStrategy
# ------------------------------------------------
# This class allows you to switch between different data inspection strategies.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.
        
        Returns:
        None: Sets the strategy for data inspection.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets the strategy for data inspection.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:    
        None
        """
        self._strategy = strategy
        
        
    def execute_strategy(self, df: pd.DataFrame):
        """
        Executes the current strategy for data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Uses the current strategy to inspect the dataframe.
        """
        self._strategy.inspect(df)
        
        
# example usage

if __name__ == "__main__":
    
    # #Read the data from the extracted_data .csv file
    # df = pd.read_csv(r"C:\Users\rohit\VSCode\End_to_End_Production_Project\prices-predictor-system\extracted_data\AmesHousing.csv")

    # #Initialize the DataInspector with the DataTypesInspectionStrategy
    # data_inspector = DataInspector(DataTypesInspectionStrategy())
    
    # #Execute the strategy to inspect the dataframe
    # data_inspector.execute_strategy(df)
    
    # #Change the strategy to SummaryStatisticsInspectionStrategy
    # data_inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    
    # #Execute the new strategy to inspect the dataframe
    # data_inspector.execute_strategy(df)
    
    pass
