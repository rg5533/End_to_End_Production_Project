import logging
from typing import Tuple
import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy

from zenml import step

@step
def data_splitter_step(df: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        target_column (str): The target column name.

    Returns:
    
    """
    splitter = DataSplitter(SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    return X_train, X_test, y_train, y_test
