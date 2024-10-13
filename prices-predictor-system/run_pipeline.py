import click
from pathlib import Path
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri



@click.command()
def main():
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    # Run the pipeline
    run = ml_pipeline()

    # You can uncomment and customize the following lines if you want to retrieve and inspect the trained model:
    # trained_model = run["model_building_step"]  # Replace with actual step name if different
    # print(f"Trained Model Type: {type(trained_model)}")

    # Get the tracking URI from ZenML stack
    tracking_uri = get_tracking_uri()

    # Debug: Print the original tracking URI
    print(f"Original tracking URI: {tracking_uri}")

    # Check if the tracking_uri starts with 'file:'
    if tracking_uri.startswith("file:"):
        # Remove the 'file:' prefix
        path_str = tracking_uri[5:]
    else:
        path_str = tracking_uri

    # Convert backslashes to forward slashes for Windows paths
    path_str = path_str.replace('\\', '/')

    # Create a Path object and resolve to an absolute path
    tracking_path = Path(path_str).resolve()

    # Convert the Path object to a proper URI
    tracking_uri_corrected = tracking_path.as_uri()

    # Debug: Print the corrected tracking URI
    print(f"Corrected tracking URI: {tracking_uri_corrected}")

    # Print the MLflow UI command with the corrected URI
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri \"{tracking_uri_corrected}\"\n"
        "To inspect your experiment runs within the MLflow UI.\n"
        "You can find your runs tracked within the experiment."
    )


if __name__ == "__main__":
    main()
