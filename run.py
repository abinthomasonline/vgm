import time

import dask.array as da
import dask.dataframe as dd
import pandas as pd


def original():
    print("Original Implementation")

    from data_transformer import DataTransformer

    train_data = pd.read_csv("assets/Credit.csv")[["Amount"]]
    transformer = DataTransformer(train_data=train_data)

    start_time = time.time()
    transformer.fit()
    end_time = time.time()
    print(f"Time taken to fit: {end_time - start_time} seconds")

    start_time = time.time()
    transformed_train_data = transformer.transform(train_data.values)
    inverse_transformed_train_data = transformer.inverse_transform(
        transformed_train_data
    )
    regen_df = pd.DataFrame(inverse_transformed_train_data, columns=["Amount"])
    print((train_data["Amount"] - regen_df["Amount"]).abs().describe())
    end_time = time.time()
    print(
        f"Time taken to transform and inverse transform: {end_time - start_time} seconds"
    )


def distributed():
    print("Distributed Implementation")

    from data_transformer import DistributedDataTransformer

    train_data = dd.read_csv("assets/Credit.csv")[["Amount"]]
    transformer = DistributedDataTransformer(train_data=train_data, verify_mode=True)

    start_time = time.time()
    transformer.fit()
    end_time = time.time()
    print(f"Time taken to fit: {end_time - start_time} seconds")

    start_time = time.time()
    transformed_train_data = transformer.transform(
        train_data.to_dask_array(lengths=True)
    )
    inverse_transformed_train_data = transformer.inverse_transform(
        transformed_train_data
    )
    regen_df = dd.from_dask_array(inverse_transformed_train_data, columns=["Amount"])
    print(
        dd.from_dask_array(
            da.abs(
                train_data.to_dask_array(lengths=True)
                - regen_df.to_dask_array(lengths=True)
            )
        )[0]
        .describe()
        .compute()
    )
    end_time = time.time()
    print(
        f"Time taken to transform and inverse transform: {end_time - start_time} seconds"
    )


def main():
    original()
    distributed()


if __name__ == "__main__":
    main()
