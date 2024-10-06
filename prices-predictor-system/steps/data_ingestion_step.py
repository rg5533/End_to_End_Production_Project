import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """
    Ingest data from a zip file and return a pandas DataFrame.
    """
    #get the file extension
    file_extension = ".zip"
    ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = ingestor.ingest_data(file_path)
    return df

