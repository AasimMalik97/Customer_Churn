from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from customer_churn.config import FIGURES_DIR, PROCESSED_DATA_DIR
import pandas as pd
import matplotlib.pyplot as plt


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "../data/processed/dataset.csv",
    output_path: Path = FIGURES_DIR / "../data/interim/plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----

    # Load the dataset
    data = pd.read_csv(input_path)

    # Generate a simple plot (e.g., histogram of a column named 'feature')
    plt.figure(figsize=(10, 6))
    plt.hist(data['feature'], bins=30, alpha=0.75, color='blue')
    plt.title('Histogram of Feature')
    plt.xlabel('Feature')
    plt.ylabel('Frequency')

    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()
    # -----------------------------------------


if __name__ == "__main__":
    app()
