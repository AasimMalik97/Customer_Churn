from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import typer
from loguru import logger
from tqdm import tqdm
import mlflow
import mlflow.sklearn

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from customer_churn.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    # Load the dataset
    df = pd.read_csv(features_path)

    # Define features and target variable
    X = df.drop(columns=['Churn', 'CustomerID'])
    y = df['Churn']

    # Handle categorical variables and missing values
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.mean())

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow run
    with mlflow.start_run():
        # Initialize and train the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Classification Report:\n{report}")

        # Log parameters and metrics to MLflow
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")

        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save the model locally
        joblib.dump(model, model_path)
        logger.success(f"Model saved to {model_path}")

if __name__ == "__main__":
    app()