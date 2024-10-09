import pandas as pd
from src.feature_engineering import (
    FeatureEngineer, 
    LogTransformation, 
    StandardScaling, 
    MinMaxScaling, 
    OneHotEncoding
)
from zenml import step

@step
def feature_engineering_step(df: pd.DataFrame, strategy: str = "log", features: list = None) -> pd.DataFrame:
    """
    Apply feature engineering techniques to the dataframe
    """
    if strategy == "log":
        strategy = FeatureEngineer(strategy=LogTransformation(features=features))
    elif strategy == "standard_scaling":
        strategy = FeatureEngineer(strategy=StandardScaling(features=features))
    elif strategy == "minmax_scaling":
        strategy = FeatureEngineer(strategy=MinMaxScaling(features=features))
    elif strategy == "onehot_encoding":
        strategy = FeatureEngineer(strategy=OneHotEncoding(features=features))
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    transformed_df = strategy.apply_feature_engineering(df)
    return transformed_df
    

