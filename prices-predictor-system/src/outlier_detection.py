import logging
from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



#Abstract base class for outlier detection
#----------------------------------------
class OutlierDetectionStrategy:
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """_summary_
            Abstract method to detect outliers in the dataframe
        Args:
            df (pd.DataFrame): The dataframe to detect outliers in

        Returns:
            pd.DataFrame: _description_
        """
        pass
    


#Concrete strategy for Z-score method
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold= 3):
        self.threshold = threshold
        
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using Z-score method")
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info(f"Outliers detected with Z-score threshold: {self.threshold}.")
        outlier_df = pd.DataFrame()
        return outliers

# Concrete Strategy for IQR Based Outlier Detection
class IQRBasedOutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using IQR method")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
        logging.info(f"Outliers detected using IQR method.")
        return outliers
    
    
#Context class for outlier detection
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy
        
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._strategy.detect_outliers(df)
    
    def handle_outliers(self, df: pd.DataFrame, method = "remove", **kwargs):
        outliers = self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing outliers from the dataset.")
            df_cleaned = df[(~outliers).all(axis=1)]        
        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            raise ValueError(f"Invalid method: {method}. No outlier handling performed.")
            return df

        logging.info("Outlier handling completed.")
        return df_cleaned

            
    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info("Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")
    

# Example usage
if __name__ == "__main__":
    # # Example dataframe
    # df = pd.read_csv("../extracted_data/AmesHousing.csv")
    # df_numeric = df.select_dtypes(include=[np.number]).dropna()

    # # Initialize the OutlierDetector with the Z-Score based Outlier Detection Strategy
    # outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))

    # # Detect and handle outliers
    # outliers = outlier_detector.detect_outliers(df_numeric)
    # df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")

    # print(df_cleaned.shape)
    # # Visualize outliers in specific features
    # # outlier_detector.visualize_outliers(df_cleaned, features=["SalePrice", "Gr Liv Area"])
    pass
