from typing import Callable, List, Dict

import pandas as pd
import pytest
import numpy as np

from pythonlatex.highlight_formatter import HighlightFormatter
from pythonlatex.pandas_to_latex import PandasTable


def dummy_highlight_method(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Dummy highlight method for testing.
    Returns the closest value to the mean of each column.
    """
    idx_closest = (df - df.mean()).abs().idxmin().to_frame()

    def parse(row):
        idx = row.values[0]
        if isinstance(idx, tuple):
            idx = idx[0]
        if not isinstance(idx, str) and np.isnan(idx):
            return np.nan
        return df.loc[idx, row.name]

    return idx_closest.apply(parse, axis=1)


@pytest.fixture(
    params=["min", "max", dummy_highlight_method],
)
def highlight_method(request):
    """Fixture to provide different highlight methods."""
    return request.param


@pytest.fixture(
    params=[
        ["\\textbf{"],
        ["\\textbf{\\underline{", "\\textbf{"],
        ["\\textbf{\\underline{", "\\textbf{", "Wow: "],
    ]
)
def highlight_format(request):
    """Fixture to provide different highlight formats."""
    return request.param


@pytest.fixture()
def formatter(
    highlight_method: Callable[[pd.DataFrame], pd.Series] | str,
    highlight_format: List[str],
) -> HighlightFormatter:
    return HighlightFormatter(
        highlight_method=highlight_method,
        highlight_format=highlight_format,
    )


@pytest.fixture(params=[2, 4, 8])
def n_rows(request):
    return request.param


@pytest.fixture(params=[8, 16, 32])
def n_cols(request):
    return request.param


@pytest.fixture(params=["random", "randint"])
def sampling_method(request):
    return request.param


@pytest.fixture()
def df_base(n_rows: int, n_cols: int, sampling_method: str) -> pd.DataFrame:
    rows = [
        "row_{}".format(i)
        for i in range(n_rows)
        for _ in range(n_cols)
        for _ in range(3)
    ]  # 10 observations
    cols = [
        "col_{}".format(j)
        for i in range(n_rows)
        for j in range(n_cols)
        for _ in range(3)
    ]
    if sampling_method == "random":
        values = np.random.random((n_rows * n_cols * 3,))
    elif sampling_method == "randint":
        values = np.random.randint(0, 5, (n_rows * n_cols * 3,)) / 5
    df = pd.DataFrame(
        {
            "rows": rows,
            "cols": cols,
            "values": values,
        }
    )

    return df


@pytest.fixture()
def df(df_base: pd.DataFrame) -> pd.DataFrame:
    df = df_base.pivot_table(index="rows", columns="cols", values="values")
    return df


@pytest.fixture(params=[0, 1, 2])
def multi_cols(request, df: pd.DataFrame) -> Dict[str, List[str]] | None:
    n_levels = request.param
    if n_levels == 0:
        return None
    multi_cols = {}
    if n_levels > 0:
        for i in range(n_levels):
            for j, c in enumerate(df.columns.unique()):
                if c not in multi_cols:
                    multi_cols[c] = [c]
                multi_cols[c] = [f"level_{i}_{j%(2**(i+1))}"] + multi_cols[c]
    return multi_cols


@pytest.fixture()
def df_multi_cols(df: pd.DataFrame, multi_cols: Dict[str, List[str]]) -> pd.DataFrame:
    if multi_cols is not None:
        df.columns = pd.MultiIndex.from_tuples(
            [tuple(multi_cols[df.columns[i]]) for i in range(len(df.columns))]
        )
    return df


@pytest.fixture()
def values_to_highlight(
    df: pd.DataFrame, highlight_method: Callable[[pd.DataFrame], pd.Series] | str
) -> pd.Series:
    if callable(highlight_method):
        return highlight_method(df)
    if highlight_method == "max":
        return df.max()
    if highlight_method == "min":
        return df.min()
    raise ValueError(highlight_method)


def test_highlight_formatter_get_vals_to_highlight(
    formatter: HighlightFormatter, df: pd.DataFrame, values_to_highlight: pd.Series
):
    v = formatter._get_vals_to_highlight(df)
    assert (v == values_to_highlight).all()


def test_highlight_formatter_get_all_masks(
    formatter: HighlightFormatter, df: pd.DataFrame, values_to_highlight: pd.Series
):
    masks = formatter._get_all_masks(df)
    total_mask: None | pd.DataFrame = None
    for i, mask in enumerate(masks):
        if i == 0:
            total_mask = mask
            for col in df.columns:
                for best in mask[mask[col]].index:
                    assert df[col][best] == values_to_highlight[col]
        else:
            total_mask = total_mask | mask
            # check that the mask is not overlapping with the previous one
            assert not (mask & masks[i - 1]).any().all()

        if i < df.shape[0]:
            # check if i values are highlighted in total
            assert (total_mask.sum() >= 1 + i).all()


def test_highlight_formatter_call(
    formatter: HighlightFormatter,
    df_multi_cols: pd.DataFrame,
    values_to_highlight: pd.Series,
):
    style = formatter(df_multi_cols)
    for col in style.columns:
        n_found = 0
        for i, fmt in enumerate(formatter.highlight_format[: style.shape[0]]):
            found = False
            for j in range(len(style[col])):
                value = style[col][j]
                if value.startswith(fmt):
                    n_found += 1
                    found = True
                    if i == 0:
                        assert value == fmt + str(values_to_highlight[col]) + "}" * (
                            fmt.count("{") - fmt.count("}")
                        )
            # Error if the formatting is not found unless we had equal values
            assert found or n_found >= i + 1, f"{fmt} not found in column {col}"


def test_pandastable(
    df_base: pd.DataFrame,
    multi_cols: None | Dict[str, List[str]],
    formatter: HighlightFormatter,
    values_to_highlight: pd.Series,
):
    table = PandasTable(
        df_base, higlighter=formatter, add_std=False, multicols=multi_cols
    )
    table_std = PandasTable(
        df_base, higlighter=formatter, add_std=True, multicols=multi_cols
    )
    df_t = table["rows", "cols", "values"]
    df_t_std = table_std["rows", "cols", "values"]
    # Remove the +- sign from the std
    for col in df_t_std.columns:
        df_t_std[col] = df_t_std[col].apply(
            lambda x: x.split("$\\pm$ \\tiny ")[0]
            + "}"
            * (
                x.split("$\\pm$ \\tiny ")[0].count("{")
                - x.split("$\\pm$ \\tiny ")[0].count("}")
            )
        )
    # Check that the values are the same
    assert (df_t == df_t_std).all().all()
