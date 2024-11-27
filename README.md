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
    - A local dask cluster should suffice for scaling, though minor vertical scaling might be needed to meet the 10-minute constraint.

3. NumPy to Dask:
    - Subclass and modify scikit-learn's BGM (Bayesian Gaussian Mixture) implementation to use dask for large array operations.
    - Adjust the transform and inverse_transform methods for compatibility with dask.

4. Inverse Transform Accuracy:
    - The research code has limitations in inverse transforming extreme values, which might be due to insufficient convergence or arbitrary scaling constant "4". These need tuning.

5. Alternate Approach:
    - Using only a subset of data for model fitting is simpler but doesn't align with the task's goals.
    

## Prioritization and Progress

1. Completed:
    - Implemented Dask-compatible Bayesian GMM (BGM).
    - Modified DataTransformer methods to work with Dask.
    - Ensured consistency with the original research code.

2. In Progress:
    - Reading and processing 1B records within 10 minutes.

3. Pending:
    - Tuning parameters and hyperparameters to handle extreme values in inverse transformations accurately.


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
