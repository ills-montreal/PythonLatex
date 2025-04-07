"""Uses seaborn datasets to test the PandasTableFormatter class."""

from typing import Tuple, Union, Callable

from itertools import combinations

import pytest

import numpy as np
import pandas as pd
import seaborn as sns

from pythonlatex.pandas_to_latex import PandasTableFormatter


FUNCS_HIGHLIGHT = [
    np.nanmin,
    np.nanmax,
    lambda x: np.nanmin(x - np.nanmean(x)) + np.nanmean(x),
]

BASE_METHODS = ["mean", "median", "std", "min", "max"]
AGG_METHODS = [
    list(c) for k in range(1, 4) for c in list(combinations(BASE_METHODS, k))[::100]
]

SEABORN_DATA_NAMES = ["dots", "mpg", "penguins", "taxis", "titanic"]


@pytest.fixture(scope="class", params=SEABORN_DATA_NAMES)
def seaborn_data(
    request,
) -> Tuple[pd.DataFrame, list[str], list[str], str]:
    """Fixture to get a seaborn dataset."""
    name = request.param
    df = sns.load_dataset(name)
    # Rows are all the columns that are of type object
    rows = df.select_dtypes(include=["object"]).columns.tolist()
    # Columns are all the columns that are not of type object
    cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Values will be set to $INSERT_NAME$
    values = "INSERT_NAME"
    # Select 3 random columns to use, and 2 random row
    cols = list(np.random.choice(cols, 3, replace=False))
    rows = list(np.random.choice(rows, 2, replace=False))

    return df, rows, cols, values


@pytest.fixture(
    scope="class",
    params=[
        (1, 1, 1),
        (1, 1, 3),
        (1, 8, 1),
        (1, 8, 3),
        (3, 1, 1),
        (3, 1, 16),
        (3, 8, 1),
        (3, 8, 16),
    ],
)
def synthetic_dataframe(request) -> pd.DataFrame:
    """Fixture to create a synthetic dataframe."""
    n_levels, n_rows, n_cols = request.param

    # Create a synthetic dataframe with random values
    def get_one_row(n_levels: int, n_cols: int) -> pd.DataFrame:
        """Gets a synthetic dataframe with random values."""
        data = {}
        data["value"] = np.random.rand(n_cols)
        for line in range(n_levels):
            # Level is the writing in binary of the column index
            data["level_{}".format(line)] = [
                int(bin(i)[2:].zfill(n_levels)[line]) for i in range(n_cols)
            ]
        data["row"] = [0 for _ in range(n_cols)]
        return pd.DataFrame(data)

    all_rows = []
    for i_row in range(n_rows):
        for j in range(4):
            row = get_one_row(n_levels, n_cols)
            row["row"] = i_row
            all_rows.append(row)
    df = pd.concat(all_rows, ignore_index=True)
    return df


@pytest.fixture(scope="class", params=FUNCS_HIGHLIGHT)
def func_to_highlight(request) -> Callable:
    """Fixture to get the function to highlight."""
    func = request.param
    if callable(func):
        return func
    else:
        raise ValueError(f"Function {func} is not callable.")


@pytest.fixture(params=[0, 1, 2, 3], scope="class")
def n_decimals(request) -> int:
    """Fixture to get the number of decimals."""
    return request.param


@pytest.fixture(
    params=AGG_METHODS,
    scope="class",
)
def aggregation_methods(request) -> Union[str, Tuple[str, ...]]:
    """Fixture to get the aggregation methods."""
    return request.param


@pytest.mark.dependency()
def test_aggregate_results_and_pivot(synthetic_dataframe, aggregation_methods):
    """Test the aggregate_results_and_pivot method."""
    formatter = PandasTableFormatter(
        n_decimals=2,
        aggregation_methods=list(aggregation_methods),
        main_subset=0,
        hide_agg_labels=True,
        already_rotated=True,
    )

    columns = [c for c in synthetic_dataframe.columns if c.startswith("level_")]
    df_agg = formatter._aggregate_results_and_pivot(
        synthetic_dataframe,
        rows=["row"],
        cols=columns,
        values="value",
    )
    # Test the number of levels of the cols is coherent
    assert df_agg.columns.nlevels == len(columns) + 1
    # Test the name of the columns
    assert df_agg.columns.names == columns + ["agg"]

    # Value per row
    synthetic_dataframe_agg = synthetic_dataframe.groupby(["row"] + columns).agg(
        aggregation_methods
    )
    # Verify That the values are equal
    for row_agg in synthetic_dataframe_agg.index:
        row_id = row_agg[0]
        cols_id = row_agg[1:]
        for agg in aggregation_methods:
            assert (
                synthetic_dataframe_agg.loc[row_agg, ("value", agg)]
                == df_agg.loc[row_id, cols_id + (agg,)]
            )


@pytest.mark.dependency(depends=["test_aggregate_results_and_pivot"])
def test_latex_synthetic(
    synthetic_dataframe, func_to_highlight, n_decimals, aggregation_methods
):
    """Simply tests if the style method works."""

    formatter = PandasTableFormatter(
        n_decimals=n_decimals,
        aggregation_methods=aggregation_methods,
        main_subset=0,
        hide_agg_labels=False,
        already_rotated=False,
    )
    columns = [c for c in synthetic_dataframe.columns if c.startswith("level_")]

    formatter.style(
        synthetic_dataframe,
        rows=["row"],
        cols=columns,
        values="value",
        highlight_fn=func_to_highlight,
        props=["font-weight: bold;"],
    )


@pytest.mark.dependency(depends=["test_style_synthetic"])
def test_style_seaborn(
    seaborn_data,
    func_to_highlight,
    n_decimals,
    aggregation_methods,
):
    """Simply tests if the style method works."""
    df, rows, cols, values = seaborn_data
    formatter = PandasTableFormatter(
        n_decimals=n_decimals,
        aggregation_methods=aggregation_methods,
        main_subset=0,
        already_rotated=True,
    )

    formatter.style(
        df,
        rows=rows,
        cols=cols,
        values=values,
        highlight_fn=func_to_highlight,
        props=["font-weight: bold;"],
    )
