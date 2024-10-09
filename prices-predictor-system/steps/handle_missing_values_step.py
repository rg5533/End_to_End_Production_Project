import pandas as pd
from zenml import step
from src.handle_missing_values import MissingValueHandler, DropMissingValuesStrategy, FillMissingValuesStrategy

@step
def handle_missing_values_step(df: pd.DataFrame, strategy: str="mean") -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    """
    if strategy == "drop":
        handler = MissingValueHandler(strategy=DropMissingValuesStrategy(axis=0))
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValueHandler(strategy=FillMissingValuesStrategy(method=strategy))
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")
    
    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df

