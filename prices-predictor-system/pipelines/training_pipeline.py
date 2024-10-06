from steps.data_ingestion_step import data_ingestion_step
from zenml import pipeline, step, Model



@pipeline(
    model=Model(
        name="prices-predictor-model",
        description="Model to predict house prices"
    ),
)
def ml_pipeline():
    """Define an end-to-end ML pipeline"""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path=r"C:\Users\rohit\VSCode\End_to_End_Production_Project\prices-predictor-system\data\archive.zip"
    )

    #Handle missing values step


    #Feature engineering step


    #Outlier detection step


    #Data splitting step


    #Model building step


    #Model evaluation step
