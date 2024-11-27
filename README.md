## Introduction
This README explains the thought process, assumptions, steps, and code structure. The goal of the task is to scale the provided research code to handle 1B rows of data efficiently and ensure it is production-ready.

## Observations and Assumptions

1. Reading the Data:
    - The provided dataset with ~50k rows and a single "Amount" column is ~0.27MB. Extrapolated, 1B rows would be <6GB, which can fit in memory on modern systems.
    - However, fitting BGM involves creating arrays (e.g., the responsibility matrix), significantly increasing memory usage.
    - CSV is a row-wise format, so reading the full dataset is required to extract "Amount".
    - Tools like Dask or Spark can handle this lazily.

2. Scaling Considerations:
    - Scaling the research code is a priority. The model must be fit and the data transformed without loading everything into memory at once.
    - Dask is a natural choice due to its compatibility with numpy and suitability for data that isn't in the TB range.
    - Using only a tiny subset of data for model fitting is simpler but doesn't align with the task's goals.
    - Fitting multiple models in parallel and then combining the weights is also simpler but doesn't align with the task's goals.

3. NumPy to Dask:
    - Subclass and modify scikit-learn's BGM (Bayesian Gaussian Mixture) implementation to use dask for large array operations.
    - Adjust the transform and inverse_transform methods for compatibility with dask.

4. Inverse Transform Accuracy:
    - The research code has limitations in inverse transforming extreme values, which might be due to insufficient convergence or arbitrary scaling constant "4". These need tuning.

    

## Prioritization and Progress

1. Completed:
    - Implemented Dask-compatible Bayesian GMM (BGM).
        - `from dbgm import DBGM as BayesianGaussianMixture`
    - Modified DataTransformer methods to work with Dask.
        - `from data_transformer import DistributedDataTransformer`
    - Ensured consistency with the original research code.
        - `pytest tests/test_transformers.py -v`
    - Reading and processing 1B records within 10 minutes.
        - Status on local cluster with 4 workers, 2 threads per worker and 16GB memory limit:
            - **~7mins** to transform and inverse transform **1B rows** of data.
            - **~5mins** to fit the BGM model on **10M rows** of data.
2. Pending:
    - Tuning parameters and scaling constant to handle extreme values in inverse transformations accurately.
    - Data validation.
    - Bindings for remote clusters and cloud storage.


## Notes on Correctness and Consistency
1. Random Generators:
    - numpy's and dask's random generators are not the same. For consistency during testing, numpy's generator is used in dask operations.

2. Responsibility Matrix Initialization:
    - BGM uses kmeans for initialization, which can vary depending on the initial cluster assignments.
    - The initialization strategy in scikit-learn is different than dask-ml.
    - To ensure consistency, scikit-learn's kmeans is used instead of dask-ml's implementation during testing.

## Code Structure

`assets/`: Data, research notebook and research paper.

`data_transformer/`: Original and Distributed versions of the DataTransformer class.

`dbgm/`: Dask-compatible BGM implementation.

`tests/`: Tests for consistency and correctness.

`requirements.txt`: Required packages.

`run.ipynb`: Jupyter notebook for demo.

`run.py`: Quick script to run the model fitting and transformation.


## How to Run the Code

1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the Jupyter notebook `run.ipynb` for a demonstration of the code.

3. Run tests using:
    ```bash
    pytest tests/test_transformers.py -v
    ```
