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
from customer_churn.config import MODELS_DIR, PROCESSED_DATA_DIR
from sklearn.tree import DecisionTreeClassifier



import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))



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



    # Train and evaluate another model (e.g., Decision Tree)

    # Initialize and train the Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    # Make predictions with the Decision Tree model
    dt_y_pred = dt_model.predict(X_test)

    # Evaluate the Decision Tree model
    dt_accuracy = accuracy_score(y_test, dt_y_pred)
    dt_report = classification_report(y_test, dt_y_pred)

    logger.info(f"Decision Tree Accuracy: {dt_accuracy}")
    logger.info(f"Decision Tree Classification Report:\n{dt_report}")

    # Log parameters and metrics to MLflow for the Decision Tree model
    mlflow.log_param("model_type", "DecisionTree")
    mlflow.log_metric("dt_accuracy", dt_accuracy)
    mlflow.log_text(dt_report, "dt_classification_report.txt")

    # Log the Decision Tree model to MLflow
    mlflow.sklearn.log_model(dt_model, "decision_tree_model")

    # Save the Decision Tree model locally
    dt_model_path = model_path.with_name("decision_tree_model.pkl")
    joblib.dump(dt_model, dt_model_path)
    logger.success(f"Decision Tree model saved to {dt_model_path}")


    # Train and evaluate another model (e.g., Random Forest)


    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions with the Random Forest model
    rf_y_pred = rf_model.predict(X_test)

    # Evaluate the Random Forest model
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    rf_report = classification_report(y_test, rf_y_pred)

    logger.info(f"Random Forest Accuracy: {rf_accuracy}")
    logger.info(f"Random Forest Classification Report:\n{rf_report}")

    # Log parameters and metrics to MLflow for the Random Forest model
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("rf_accuracy", rf_accuracy)
    mlflow.log_text(rf_report, "rf_classification_report.txt")

    # Log the Random Forest model to MLflow
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

    # Save the Random Forest model locally
    rf_model_path = model_path.with_name("random_forest_model.pkl")
    joblib.dump(rf_model, rf_model_path)
    logger.success(f"Random Forest model saved to {rf_model_path}")