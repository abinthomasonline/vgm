import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from data_transformer import DataTransformer, DistributedDataTransformer


@pytest.fixture
def credit_data():
    """Load Credit.csv data for testing"""
    return pd.read_csv("assets/Credit.csv")[["Amount"]]


@pytest.fixture
def distributed_credit_data():
    """Load Credit.csv data as dask dataframe for testing"""
    return dd.read_csv("assets/Credit.csv")[["Amount"]]


def test_fit_weights_match(credit_data, distributed_credit_data):
    """Test if fitted model weights match between DataTransformer and DistributedDataTransformer"""

    # Initialize transformers
    transformer = DataTransformer(train_data=credit_data)
    dist_transformer = DistributedDataTransformer(
        train_data=distributed_credit_data,
        verify_mode=True,  # Enable deterministic mode
    )

    # Fit both transformers
    transformer.fit()
    dist_transformer.fit()

    # Check if weights match for the first model (Amount column)
    np.testing.assert_allclose(
        transformer.model[0].weights_,
        dist_transformer.model[0].weights_,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Model weights do not match between transformers",
    )


def test_transform_outputs_match(credit_data, distributed_credit_data):
    """Test if transform outputs match between DataTransformer and DistributedDataTransformer"""

    # Initialize and fit transformers
    transformer = DataTransformer(train_data=credit_data)
    dist_transformer = DistributedDataTransformer(
        train_data=distributed_credit_data, verify_mode=True
    )

    transformer.fit()
    dist_transformer.fit()

    # Transform data
    transformed = transformer.transform(credit_data.values)
    dist_transformed = dist_transformer.transform(
        distributed_credit_data.to_dask_array(lengths=True)
    ).compute()

    # Compare transformed outputs
    np.testing.assert_allclose(
        transformed,
        dist_transformed,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Transformed outputs do not match between transformers",
    )


def test_inverse_transform_outputs_match(credit_data, distributed_credit_data):
    """Test if inverse transform outputs match between DataTransformer and DistributedDataTransformer"""

    # Initialize and fit transformers
    transformer = DataTransformer(train_data=credit_data)
    dist_transformer = DistributedDataTransformer(
        train_data=distributed_credit_data, verify_mode=True
    )

    transformer.fit()
    dist_transformer.fit()

    # Transform data
    transformed = transformer.transform(credit_data.values)
    dist_transformed = dist_transformer.transform(
        distributed_credit_data.to_dask_array(lengths=True)
    )

    # Inverse transform
    inverse_transformed = transformer.inverse_transform(transformed)
    dist_inverse_transformed = dist_transformer.inverse_transform(
        dist_transformed
    ).compute()

    # Compare inverse transformed outputs
    np.testing.assert_allclose(
        inverse_transformed,
        dist_inverse_transformed,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Inverse transformed outputs do not match between transformers",
    )


def test_end_to_end_reconstruction(credit_data, distributed_credit_data):
    """Test if both transformers can accurately reconstruct the original data"""

    # Initialize and fit transformers
    transformer = DataTransformer(train_data=credit_data)
    dist_transformer = DistributedDataTransformer(
        train_data=distributed_credit_data, verify_mode=True
    )

    transformer.fit()
    dist_transformer.fit()

    # Original data
    original_values = credit_data.values

    # Transform and inverse transform with regular transformer
    transformed = transformer.transform(original_values)
    reconstructed = transformer.inverse_transform(transformed)

    # Transform and inverse transform with distributed transformer
    dist_transformed = dist_transformer.transform(
        distributed_credit_data.to_dask_array(lengths=True)
    )
    dist_reconstructed = dist_transformer.inverse_transform(dist_transformed).compute()

    # Compare reconstructed data with original
    np.testing.assert_allclose(
        original_values,
        reconstructed,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Regular transformer failed to reconstruct original data",
    )

    np.testing.assert_allclose(
        original_values,
        dist_reconstructed,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Distributed transformer failed to reconstruct original data",
    )
