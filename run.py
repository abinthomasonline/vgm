import argparse
import json
import logging
import time
import uuid
from datetime import datetime

import pandas as pd
import s3fs

from data_transformer import DataTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_data(input_file, output_path) -> None:
    """
    Process data using DataTransformer.

    Args:
        input_file: Path to input CSV file
        output_path: Path to output directory
    """
    time_taken = {"fit": 0, "transform": 0, "inverse_transform": 0}

    run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + str(uuid.uuid4())
    run_path = output_path + run_id

    s3 = s3fs.S3FileSystem(anon=False)
    s3.mkdir(run_path)
    s3.mkdir(run_path + "/transformed")
    s3.mkdir(run_path + "/inverse_transformed")

    logger.info("Loading data from %s", input_file)

    # Fit transformer
    start_time = time.time()
    train_data = pd.read_csv(input_file)[["Amount"]]
    transformer = DataTransformer(train_data=train_data)
    transformer.fit()
    elapsed = time.time() - start_time
    logger.info("Time taken to fit: %.2f seconds", elapsed)
    time_taken["fit"] = elapsed

    # Transform data
    start_time = time.time()
    train_data = pd.read_csv(input_file)[["Amount"]]
    transformed_train_data = transformer.transform(train_data.values)
    transformed_df = pd.DataFrame(transformed_train_data)
    transformed_df.to_csv(run_path + "/transformed/001.csv", index=False)
    elapsed = time.time() - start_time
    logger.info("Time taken to transform: %.2f seconds", elapsed)
    time_taken["transform"] = elapsed

    # Inverse transform data
    start_time = time.time()
    transformed_train_data = pd.read_csv(run_path + "/transformed/001.csv").values
    inverse_transformed_train_data = transformer.inverse_transform(
        transformed_train_data
    )
    regen_df = pd.DataFrame(inverse_transformed_train_data, columns=["Amount"])
    regen_df.to_csv(run_path + "/inverse_transformed/001.csv", index=False)
    elapsed = time.time() - start_time
    logger.info("Time taken to inverse transform: %.2f seconds", elapsed)
    time_taken["inverse_transform"] = elapsed

    # Write timing info
    with s3.open(run_path + "/timing.json", "w") as f:
        json.dump(time_taken, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Process data using DataTransformer")
    parser.add_argument(
        "--input-file",
        type=str,
        default="s3://vgm-dask/input/Credit.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="s3://vgm-dask/output/",
        help="Path to output directory",
    )
    args = parser.parse_args()

    process_data(args.input_file, args.output_path)


if __name__ == "__main__":
    main()
