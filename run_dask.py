import argparse
import json
import logging
import time
import uuid
from datetime import datetime

import dask.dataframe as dd
import s3fs
from dask.distributed import Client
from dask_kubernetes.operator import KubeCluster

from data_transformer import DistributedDataTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_data(
    input_file: str,
    output_path: str,
    cluster_spec_file: str,
    n_workers: int,
    cluster_name: str,
    service_account: str,
    image: str,
    memory: str,
    cpu: str,
    n_files: int = 4,
    n_partitions: int = 4,
) -> None:
    """
    Process data using DistributedDataTransformer on a Dask cluster.

    Args:
        input_file: Path to input CSV file
        output_path: Path to output directory
        cluster_spec_file: Path to Dask cluster specification JSON
        n_workers: Number of workers to scale the cluster to
        cluster_name: Override cluster name in spec
        service_account: Override service account in spec
        image: Override container image in spec
        memory: Override memory resource limits/requests
        cpu: Override CPU resource limits/requests
        n_files: Number of copies of input file to process in parallel
        n_partitions: Number of partitions for Dask DataFrames
    """
    time_taken = {"fit": 0, "transform": 0, "inverse_transform": 0}

    run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + str(uuid.uuid4())
    run_path = output_path + run_id

    logger.info("Loading cluster specification from %s", cluster_spec_file)
    with open(cluster_spec_file, "r") as f:
        cluster_spec = json.load(f)

    # Override cluster spec values if provided
    if cluster_name:
        cluster_spec["metadata"]["name"] = cluster_name
        cluster_spec["spec"]["scheduler"]["service"]["selector"][
            "dask.org/cluster-name"
        ] = cluster_name

    if service_account:
        cluster_spec["spec"]["worker"]["spec"]["serviceAccountName"] = service_account
        cluster_spec["spec"]["scheduler"]["spec"][
            "serviceAccountName"
        ] = service_account

    if image:
        cluster_spec["spec"]["worker"]["spec"]["containers"][0]["image"] = image
        cluster_spec["spec"]["scheduler"]["spec"]["containers"][0]["image"] = image

    if memory or cpu:
        resources = {"limits": {}, "requests": {}}
        if memory:
            resources["limits"]["memory"] = memory
            resources["requests"]["memory"] = memory
        if cpu:
            resources["limits"]["cpu"] = cpu
            resources["requests"]["cpu"] = cpu

        cluster_spec["spec"]["worker"]["spec"]["containers"][0]["resources"] = resources
        cluster_spec["spec"]["scheduler"]["spec"]["containers"][0][
            "resources"
        ] = resources

    logger.info("Creating output directories at %s", run_path)
    s3 = s3fs.S3FileSystem(anon=False)
    s3.mkdir(run_path)
    s3.mkdir(run_path + "/transformed")
    s3.mkdir(run_path + "/inverse_transformed")

    cluster = None
    client = None
    try:
        logger.info("Starting Dask cluster...")
        cluster = KubeCluster(custom_cluster_spec=cluster_spec)
        cluster.scale(n_workers)
        client = Client(cluster)
        logger.info("Dask cluster started successfully")

        logger.info("Starting fit process...")
        start_time = time.time()
        train_data = dd.read_csv([input_file for _ in range(n_files)])[
            ["Amount"]
        ].repartition(npartitions=n_partitions)
        transformer = DistributedDataTransformer(train_data)
        transformer.fit()
        elapsed = time.time() - start_time
        time_taken["fit"] = elapsed
        logger.info("Fit completed in %.2f seconds", elapsed)

        logger.info("Starting transform process...")
        start_time = time.time()
        train_data = dd.read_csv([input_file for _ in range(n_files)])[
            ["Amount"]
        ].repartition(npartitions=n_partitions)
        transformed_train_data = transformer.transform(
            train_data.to_dask_array(lengths=True)
        )
        transformed_df = dd.from_dask_array(transformed_train_data)
        transformed_df.to_csv(run_path + "/transformed/*.csv", index=False)
        elapsed = time.time() - start_time
        time_taken["transform"] = elapsed
        logger.info("Transform completed in %.2f seconds", elapsed)

        logger.info("Starting inverse transform process...")
        start_time = time.time()
        transformed_df = dd.read_csv(run_path + "/transformed/*.csv").repartition(
            npartitions=n_partitions
        )
        inverse_transformed_train_data = transformer.inverse_transform(
            transformed_df.to_dask_array(lengths=True)
        )
        regen_df = dd.from_dask_array(
            inverse_transformed_train_data, columns=["Amount"]
        )
        regen_df.to_csv(run_path + "/inverse_transformed/*.csv", index=False)
        elapsed = time.time() - start_time
        time_taken["inverse_transform"] = elapsed
        logger.info("Inverse transform completed in %.2f seconds", elapsed)

        logger.info("Writing timing information...")
        with s3.open(run_path + "/timing.json", "w") as f:
            json.dump(time_taken, f)
        logger.info("Process completed successfully")

    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise

    finally:
        logger.info("Cleaning up resources...")
        if cluster:
            cluster.close()
        if client:
            client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process data using DistributedDataTransformer"
    )
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
    parser.add_argument(
        "--cluster-spec",
        type=str,
        default="dask-clusterspec.json",
        help="Path to Dask cluster specification JSON file",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of workers to scale the cluster to",
    )
    parser.add_argument(
        "--cluster-name",
        type=str,
        default="my-dask-cluster-2",
        help="Override cluster name in spec",
    )
    parser.add_argument(
        "--service-account",
        type=str,
        default="vgm-dask",
        help="Override service account name in spec",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="730335639508.dkr.ecr.us-east-2.amazonaws.com/vgm-dask/worker-image:latest",
        help="Override container image in spec",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="4Gi",
        help="Override memory resource limits/requests (e.g. '4Gi')",
    )
    parser.add_argument(
        "--cpu",
        type=str,
        default="2000m",
        help="Override CPU resource limits/requests (e.g. '2000m')",
    )
    parser.add_argument(
        "--n-files",
        type=int,
        default=4,
        help="Number of copies of input file to process in parallel",
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=4,
        help="Number of partitions for Dask DataFrames",
    )
    args = parser.parse_args()

    process_data(
        args.input_file,
        args.output_path,
        args.cluster_spec,
        args.n_workers,
        args.cluster_name,
        args.service_account,
        args.image,
        args.memory,
        args.cpu,
        args.n_files,
        args.n_partitions,
    )


if __name__ == "__main__":
    main()
