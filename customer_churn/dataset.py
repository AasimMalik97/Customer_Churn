from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from customer_churn.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import pandas as pd

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "../data/processed/dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "../data/interim/dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # Load the dataset
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)

    # Perform some basic data processing
    logger.info("Performing basic data processing")
    df.dropna(inplace=True)  # Drop rows with missing values
    df = df[df['Churn'].notnull()]  # Ensure 'Churn' column has no null values

    # Save the processed dataset
    logger.info(f"Saving processed dataset to {output_path}")
    df.to_csv(output_path, index=False)
    # -----------------------------------------


if __name__ == "__main__":
    app()
