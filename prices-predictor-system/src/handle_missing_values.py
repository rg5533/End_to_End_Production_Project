import logging
from abc import ABC, abstractmethod
import pandas as pd

#setting up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Abstract class for handling missing values
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract class for handling missing values in a dataset.
        Parameters:
            df (pd.DataFrame): The input DataFrame containing missing values.
        Returns:
            pd.DataFrame: The DataFrame with missing values handled.
        """
        pass
    

#Concrete class for handling missing values by imputation
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        """Initialize the DropMissingValuesStrategy with specified axis and threshold.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values by dropping rows or columns based on the specified threshold.
        """
        logging.info("Handling missing values by dropping rows or columns with axis={self.axis}, and thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values have been dropped successfully")
        return df_cleaned

#Concrete Startegy for Filling Missing Values
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method: str = 'mean', fill_value=None):
        """Initialize the FillMissingValuesStrategy with specified strategy.
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values by filling them with the specified strategy.
        """
        logging.info("Filling missing values using method={self.method}")
        
        df_cleaned = df.copy()
        if self.method == 'mean':
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned
            
# Context Class for Handling Missing Values
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        """Initialize the MissingValueHandler with a strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """Set the strategy for handling missing values.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using the current strategy.
        """
        logging.info("Handling missing values.")
        return self._strategy.handle(df)
    
    if __name__ == "__main__":
        # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Initialize missing value handler with a specific strategy
    # missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0, thresh=3))
    # df_cleaned = missing_value_handler.handle_missing_values(df)

    # Switch to filling missing values with mean
    # missing_value_handler.set_strategy(FillMissingValuesStrategy(method='mean'))
    # df_filled = missing_value_handler.handle_missing_values(df)
        pass
        