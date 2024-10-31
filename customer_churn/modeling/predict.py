from pathlib import Path
import joblib
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm
from customer_churn.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    # Load the dataset
    df_features = pd.read_csv(features_path)

    # Load the model
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Make predictions
    predictions = model.predict(df_features)

    # Save the predictions
    df_predictions = pd.DataFrame(predictions, columns=["Predictions"])
    df_predictions.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    app()