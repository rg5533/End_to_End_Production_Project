from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
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
    filled_data = handle_missing_values_step(df=raw_data)

    #Feature engineering step
    engineered_data = feature_engineering_step(
        df=filled_data, strategy="log",
        features=["SalePrice", "Gr Liv Area"]
        )
    
    #Outlier detection step
    cleaned_data = outlier_detection_step(df=engineered_data, column_name="SalePrice")

    #Data splitting step
    X_train, X_test, y_train, y_test = data_splitter_step(
        cleaned_data, target_column="SalePrice"
        )
    
    #Model building step
    model = model_building_step(X_train=X_train, y_train=y_train)


    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model
