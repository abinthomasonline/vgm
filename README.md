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

5. Running on Kubernetes:
    - K8s operator seems to be the best way to manage the Dask cluster. A new cluster can be created just like any other k8s resource.
    - The client pod can be run as a separate pod, which can create the cluster on the fly and submit the jobs.
    - Need to ensure the client pod and the worker & scheduler pods have same python environment with compatible versions.
    

## Performance comparison with NumPy
- Handling Larger-than-Memory Data
    The primary motivation for transitioning from NumPy to Dask is scalability. NumPy operates in-memory, making it unsuitable for datasets that exceed the available system memory. Dask, on the other hand, enables distributed computations, allowing it to handle datasets much larger than memory.
- Speed Comparison
    - For computations that fit into memory, NumPy and Dask have comparable speeds.
    - Dask may exhibit slightly slower performance due to communication overhead between workers, especially in distributed setups.
- Real-World Use Cases
    - In scenarios where memory is not a constraint and the machine has sufficient resources, NumPy will typically outperform Dask due to its simplicity and lack of overhead.
    - However, for real-world problems involving large datasets, where NumPy would fail due to memory constraints, Dask provides a scalable and practical solution by distributing the workload across multiple workers or machines.


## Notes on Correctness and Consistency
1. Random Generators:
    - numpy's and dask's random generators are not the same. For consistency during testing, numpy's generator is used in dask operations.

2. Responsibility Matrix Initialization:
    - BGM uses kmeans for initialization, which can vary depending on the initial cluster assignments.
    - The initialization strategy in scikit-learn is different than dask-ml.
    - To ensure consistency, scikit-learn's kmeans is used instead of dask-ml's implementation during testing.


## How to Run the Code (EKS)
1. Install [Dask Kubernetes Operator](https://kubernetes.dask.org/en/latest//) in the k8s cluster:
    ```bash
    helm install --repo https://helm.dask.org --create-namespace -n dask-operator --generate-name dask-kubernetes-operator
    ```

2. Build the docker images and push to ECR:
    ```bash
    docker build --platform linux/amd64 -t vgm-dask/client-image .
    docker tag vgm-dask/client-image:latest 730335639508.dkr.ecr.us-east-2.amazonaws.com/vgm-dask/client-image:latest
    docker push 730335639508.dkr.ecr.us-east-2.amazonaws.com/vgm-dask/client-image:latest

    docker build --platform linux/amd64 -t vgm-dask/worker-image -f Dockerfile_dask .
    docker tag vgm-dask/worker-image:latest 730335639508.dkr.ecr.us-east-2.amazonaws.com/vgm-dask/worker-image:latest
    docker push 730335639508.dkr.ecr.us-east-2.amazonaws.com/vgm-dask/worker-image:latest
    ```

3. Apply the Kubernetes manifests to run the code:
    ```bash
    kubectl apply -f run.yaml         # Research code run
    kubectl apply -f run-dask.yaml    # Distributed run
    ```

4. Run tests using:
    ```bash
    pytest tests/test_transformers.py -v
    ```


## Prioritization and Progress
1. Completed:
    - Implemented Dask-compatible Bayesian GMM (BGM).
        - `from dbgm import DBGM as BayesianGaussianMixture`
    - Modified DataTransformer methods to work with Dask.
        - `from data_transformer import DistributedDataTransformer`
    - Ensured consistency with the original research code.
        - `pytest tests/test_transformers.py -v`
    - Reading and processing 1B records within 10 minutes.
        - Status on local cluster with **4 workers**, **2 threads per worker** and **16GB memory limit**:
            - **~7mins** to transform and inverse transform **1B rows** of data.
            - **~5mins** to fit the BGM model on **10M rows** of data.
    - Bindings for remote clusters and cloud storage.
        - Added scripts, manifests, and dockerfiles for running on AWS EKS. (Tested)
2. Pending:
    - Tuning parameters and scaling constant to handle extreme values in inverse transformations accurately.
    - Data validation.
    - GitHub Actions for testing, building docker images etc.


## Code Structure
```bash
.
├── assets/                  # Data, research notebook and research paper
├── data_transformer/       # Original and Distributed versions of the DataTransformer class
├── dbgm/                  # Dask-compatible BGM implementation
├── tests/                # Tests for consistency and correctness
├── Dockerfile           # Docker image for the application code
├── Dockerfile_dask     # Docker image for the Dask worker and scheduler
├── requirements.txt   # Required packages
├── run-dask.yaml    # Kubernetes manifest for the distributed run
├── run_dask.py     # Entrypoint for the distributed run
├── run.ipynb      # Jupyter notebook for demo
├── run.py        # Entrypoint for the research code run
└── run.yaml     # Kubernetes manifest for the research code run
```
