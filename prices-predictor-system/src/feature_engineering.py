import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

#setup logging configurations
logging.basicConfig(level=logging.info, format='%(asctime)s - %(levelname)s - %(message)s')

#Abstract class for feature engineering strategy
#-----------------------------------------
#This class defines a common interface for all feature engineering strategies.
#Subclasses must implement the apply_transform method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to apply transformation to the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe to apply transformation
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        pass
    

#Concrete feature engineering strategy for log transformation
#----------------------------------------------------------------
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """Initialize the LogTransformation with the specified features
        
        Args:
            features (list): List of feature names to apply log transformation
        """
        self.features = features
        
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to the dataframe
            
        Args:
            df (pd.DataFrame): Input dataframe to apply transformation
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        logging.info("Applying log transformation to features: {self.features}")
        
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
                )#log1p is used to avoid issues with zero values
        logging.info("Log transformation completed")
        return df_transformed

#Concrete feature engineering strategy for Standard Scaling
#----------------------------------------------------------------
#This strategy applies standard scaling to the specified features
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """Initialize the StandardScaling with the specified features
        
        Args:
            features (list): List of feature names to apply standard scaling
        """
        self.features = features
        self.scaler = StandardScaler()
        
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaling to the dataframe using scikit-learn's StandardScaler
        
        Args:
            df (pd.DataFrame): Input dataframe to apply transformation
            
        Returns:
            pd.DataFrame: Transformed dataframe with scaled features
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        
        df_transformed = df.copy()
        # Fit the scaler on the specified features and transform them
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        
        logging.info("Standard scaling completed using StandardScaler")
        return df_transformed

#Concrete feature engineering strategy for Min-Max Scaling
#----------------------------------------------------------------
#This strategy applies min-max scaling to the specified features
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """Initialize the MinMaxScaling with the specified features
        
        Args:
            features (list): List of feature names to apply min-max scaling
        """
        self.features = features
        self.scaler = MinMaxScaler()
        
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max scaling to the dataframe using scikit-learn's MinMaxScaler
        
        Args:
            df (pd.DataFrame): Input dataframe to apply transformation
            
        """
        logging.info(f"Applying min-max scaling to features: {self.features}")
        
        df_transformed = df.copy()
        # Fit the scaler on the specified features and transform them
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        
        logging.info("Min-max scaling completed using MinMaxScaler")
        return df_transformed

#Concrete feature engineering strategy for One-Hot Encoding
#----------------------------------------------------------------
#This strategy applies one-hot encoding to the specified categorical features
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """Initialize the OneHotEncoding with the specified features
        
        Args:
            features (list): List of feature names to apply one-hot encoding
        """
        self.features = features
        self.encoder = OneHotEncoder()
        
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding to the dataframe using pandas
        
        Args:
            df (pd.DataFrame): Input dataframe to apply transformation
            
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        
        df_transformed = df.copy()
        # Fit the encoder on the specified features and transform them
        encoded_features = self.encoder.fit_transform(df[self.features]).toarray()
        # Get the feature names
        feature_names = self.encoder.get_feature_names_out(self.features)
        # Create a DataFrame with the encoded features
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
        #Drop the original categorical features
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        # Concatenate the encoded features with the original dataframe
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        
        logging.info("One-hot encoding completed using OneHotEncoder")
        return df_transformed

#Context class for feature engineering
#-----------------------------------------
#This class uses the strategy pattern to apply a sequence of feature engineering strategies to the dataframe
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """Initialize the FeatureEngineer with the specified strategy
        
        Args:
            strategy (FeatureEngineeringStrategy): The strategy to apply for feature engineering
        """
        self._strategy = strategy
        
    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """Set the feature engineering strategy to be used
        
        Args:
            strategy (FeatureEngineeringStrategy): The strategy to apply for feature engineering
        """
        self._strategy = strategy
        
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the feature engineering strategy to the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe to apply transformation
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        return self._strategy.apply_transformation(df)
    
    
    if __name__ == "__main__":
        #Example usage of the feature engineering pipeline
        # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Log Transformation Example
    # log_transformer = FeatureEngineer(LogTransformation(features=['SalePrice', 'Gr Liv Area']))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)

    # Standard Scaling Example
    # standard_scaler = FeatureEngineer(StandardScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_standard_scaled = standard_scaler.apply_feature_engineering(df)

    # Min-Max Scaling Example
    # minmax_scaler = FeatureEngineer(MinMaxScaling(features=['SalePrice', 'Gr Liv Area'], feature_range=(0, 1)))
    # df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)

    # One-Hot Encoding Example
    # onehot_encoder = FeatureEngineer(OneHotEncoding(features=['Neighborhood']))
    # df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)

        pass
        
