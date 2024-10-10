import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

#configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Abstract base class for DataSplitterStrategy
#--------------------------------------------
class DataSplitterStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column:str):
        """Split the dataset into training and testing sets.

        Args:
            df (pd.DataFrame): The input DataFrame to split.
            target_column (str): The target column name.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing data.
        """
        pass
    

#Concrete strategy for train-test split
#--------------------------------
class SimpleTrainTestSplitStrategy(DataSplitterStrategy):
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        
    def split_data(self, df: pd.DataFrame, target_column: str):
        """Split the dataset into training and testing sets.

        Args:
            df (pd.DataFrame): The input DataFrame to split.
            target_column (str): The target column name.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing data.
        """
        logging.info("Splitting data into training and testing sets.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        logging.info("Training and testing data split successfully.")
        return X_train, X_test, y_train, y_test
    
# Context Class for Data Splitting
# --------------------------------
# This class uses a DataSplittingStrategy to split the data.
class DataSplitter:
    def __init__(self, strategy: DataSplitterStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy: DataSplitterStrategy):
        self._strategy = strategy
        
    def split(self, df: pd.DataFrame, target_column: str):
        return self._strategy.split_data(df, target_column)
    

# Example usage
if __name__ == "__main__":
    # Example dataframe (replace with actual data loading)
    # df = pd.read_csv('your_data.csv')

    # Initialize data splitter with a specific strategy
    # data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    # X_train, X_test, y_train, y_test = data_splitter.split(df, target_column='SalePrice')

    pass
