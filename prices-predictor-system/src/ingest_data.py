from abc import ABC, abstractmethod
import zipfile
import os
import pandas as pd


# Step 1: Define the Data Ingestor Interface
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Method to ingest data."""
        pass

class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame."""
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")
        
        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")
            
        # Find the extracted CSV file (assuming there is one CSV file inside the zip)
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Please specify which one to use.")

        # Read the CSV into a DataFrame
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        # Return the DataFrame
        return df

# Step 3: Implement the Factory Class
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on file extension."""
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")

if __name__ == "__main__":
    # #specify the file path
    # file_path = r"C:\Users\rohit\VSCode\End_to_End_Production_Project\prices-predictor-system\data\archive.zip"
    
    # #determine the file extension
    # file_extension = os.path.splitext(file_path)[1]
    
    # #get the data ingestor
    # data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    
    # #ingest the data and load it into a DataFrame   
    # df = data_ingestor.ingest(file_path)
    
    # # display the first few rows of the DataFrame
    # print(df.head())
    
    pass
    
