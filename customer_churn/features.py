from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from customer_churn.config import PROCESSED_DATA_DIR
import pandas as pd

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "../data/processed/dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "../data/interim/features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # Load the dataset
    df = pd.read_csv(input_path)

    # Example feature engineering: Adding a new column 'total_charges' as product of 'tenure' and 'monthly_charges'
    df['total_charges'] = df['tenure'] * df['monthly_charges']

    # Save the new dataframe with features
    df.to_csv(output_path, index=False)
    # -----------------------------------------


if __name__ == "__main__":
    app()
