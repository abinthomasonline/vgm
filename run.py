import time

import pandas as pd


def main():
    print("Original Implementation")

    from data_transformer import DataTransformer

    train_data = pd.read_csv("assets/Credit.csv")[["Amount"]]
    transformer = DataTransformer(train_data=train_data)

    start_time = time.time()
    transformer.fit()
    end_time = time.time()
    print(f"Time taken to read and fit: {end_time - start_time} seconds")

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


if __name__ == "__main__":
    main()
