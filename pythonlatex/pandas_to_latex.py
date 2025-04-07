from functools import partial
from typing import Union, List, Callable, Dict, Any
import numpy as np

import pandas as pd
from pandas.io.formats.style import Styler


class PandasTableFormatter:
    def __init__(
        self,
        n_decimals: int = 3,
        aggregation_methods: List[Any] = ["mean", "std"],
        main_subset: int = 0,
        total_col_name: str = "AVG.",
        hide_agg_labels: bool = True,
    ):
        """
        PandasTableFormatter is a class that formats a Pandas DataFrame into a LaTeX
        table with custom aggregation methods and styles.

        The 'main_subset' parameter is used to specify which aggregation methods should be used
        to compare the different rows in the table (often the mean). These aggregagtion results
        will be the one highlighted in the latex table.

        If main_subset is specified, a column ($NAME, agg$) will be created with the aggregation method
        """
        self.n_decimals = n_decimals
        self.aggregation_methods = aggregation_methods
        self.hide_agg_labels = hide_agg_labels

        for agg in self.aggregation_methods:
            if not isinstance(agg, str) and not callable(agg):
                raise ValueError(
                    "Aggregation methods must be either a string or a callable function"
                )
            elif callable(agg):
                if not hasattr(agg, "__name__"):
                    raise ValueError(
                        "Aggregation function must have a __name__ attribute"
                    )
        assert (
            0 <= main_subset < len(aggregation_methods)
        ), "main_subset must be an integer between 0 and the number of aggregation methods"

        self.main_subset = main_subset
        if isinstance(aggregation_methods[main_subset], str):
            self.main_agg = aggregation_methods[main_subset]
        elif hasattr(aggregation_methods[main_subset], "__main__"):
            self.main_agg = aggregation_methods[main_subset].__name__
        self.total_col_name = total_col_name

    def _find_k_th_fn(
        self,
        s: np.ndarray | pd.Series,
        fn: Callable[[np.ndarray | pd.Series], float],
        k: int,
        props="",
    ) -> List[str]:
        """
        Highlight the top k values in a dataframe using the prop string.

        :param s: The input array or series to be processed.
        :param fn: The function to be applied to the input.
        :param k: The number of top values to highlight.
        :param props: The properties to be applied to the highlighted values.

        :return: A list of strings with the highlighted values.
        """
        ps = np.where(s == fn(s), True, False)
        for _ in range(1, k):
            ps = np.where(s == fn(s[~ps]), True, False)
        return [props if p else "" for p in ps]

    def _aggregate_results_and_pivot(
        self,
        df_base: pd.DataFrame,
        rows: str | List[str],
        cols: str | List[str],
        values: str,
    ) -> pd.DataFrame:
        """
        Aggregates the given dataframe by computing the mean and standard deviation for a specified
        metric, organizing the results in a structured format.

        Example with no multicols:
            df_base = pd.DataFrame(
                {
                    "name": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
                    "category": ["A", "A", "A", "B", "B", "B", "A", "A", "A", "B", "B", "B"],
                    "value": [0,1,2,-1,0,1,-1,1,3,-10,0,10],
                }
            )
            self._aggregate_results_with_std(df_base)

            >> Output:
                 avg mean   avg std   A mean    A std    B mean      B std
            a     0.5        0.7        1         1        0          0.5
            b     0.5        0.7        1         2        0          10

        Example with multicols {"A": ["category1", "A"], "B": ["category1", "B"], "C": ["category2", "C"]}:
            df_base = pd.DataFrame(
                {
                    "name": ["a", "a", "a", "a", "b", "b", "b", "b"],
                    "category": ["A", "A", "B", "B", "A", "A", "B", "B"],
                    "value": [0,1,2,2,0,1,2,2],
                }
            )
            self._aggregate_results_with_std(df_base)

            >> Output:
                 avg    avg   A    A    B    B
                mean   std  mean  std  mean std
            a    0.5   0.5   1    1    2    0
            b    0.5   0.5   1    1    2    0


        :param df_base: The base dataframe containing the data to be aggregated.
        :param rows: The column(s) to be used as rows in the resulting dataframe.
        :param cols: The column(s) to be used as columns in the resulting dataframe.
        :param values: The column(s) to be used as values in the resulting dataframe.
        :param aggregation_methods: The aggregation functions to be applied to the values.

        :return:  A formatted dataframe containing aggregated mean and standard
        deviation values.
        """
        # Join the mean and std dataframes to a new one
        dataframes_to_concatenate = []
        df_glob = []
        for i_agg_meth, agg in enumerate(self.aggregation_methods):
            df_agg = df_base.pivot_table(
                index=rows, columns=cols, values=values, aggfunc=agg
            )
            df_agg.columns = pd.MultiIndex.from_arrays(
                [df_agg.columns.get_level_values(i) for i in range(len(cols))]
                + [
                    pd.Index(
                        [agg if isinstance(agg, str) else agg.__name__]
                        * df_agg.shape[1],
                        name="agg",
                    )
                ],
                names=cols + ["agg"] if isinstance(cols, list) else [cols, "agg"],
            )

            if i_agg_meth == self.main_subset:
                for agg_b in self.aggregation_methods:
                    df_glob_agg = df_agg.agg(agg_b, axis=1)
                    df_glob_agg = df_glob_agg.to_frame(
                        name=agg_b if isinstance(agg_b, str) else agg_b.__name__
                    )
                    df_glob_agg.columns = pd.MultiIndex.from_arrays(
                        [
                            pd.Index(
                                [" " if i < len(cols) - 1 else self.total_col_name],
                                name=cols[i],
                            )
                            for i in range(len(cols))
                        ]
                        + [
                            pd.Index(
                                [agg_b if isinstance(agg_b, str) else agg_b.__name__],
                                name="agg",
                            )
                        ],
                        names=(
                            cols + ["agg"] if isinstance(cols, list) else [cols, "agg"]
                        ),
                    )
                    df_glob.append(df_glob_agg)
            dataframes_to_concatenate.append(df_agg)
        df_agg = pd.concat(dataframes_to_concatenate, axis=1)
        df_glob = pd.concat(df_glob, axis=1)

        df_agg = pd.concat([df_agg, df_glob], axis=1)
        df_agg.index.name = None
        df_agg = df_agg.reindex(
            sorted(df_agg.columns, key=lambda x: tuple(x[:-1])),
            axis=1,
        )
        return df_agg

    def style(
        self,
        df: pd.DataFrame,
        rows: str | List[str],
        cols: str | List[str],
        values: str,
        highlight_fn: Callable[[np.ndarray | pd.Series], float] = np.nanmax,
        props: List[str] = ["font-weight: bold;"],
        special_format_agg: Dict[str, Callable[[str], str]] = {
            "std": lambda x: "\\tiny $\\pm$" + x
        },
    ) -> Styler:
        """
        Applies the highlight method to the given dataframe and returns a styled dataframe.

        :param df: The dataframe to be styled.
        :return: A styled dataframe with highlighted values.
        """
        k = len(props)
        df_agg = self._aggregate_results_and_pivot(
            df,
            rows=rows,
            cols=cols,
            values=values,
        )

        # Trick as there is a data leakage (pandas issue)
        def wrap_special_format_agg(fn: Callable[[str], str]) -> Callable[[float], str]:
            def wrapped(x: float) -> str:
                x_str = str(np.round(x, self.n_decimals))
                out = fn(x_str)
                return out

            return wrapped

        formatter = {
            c: wrap_special_format_agg(special_format_agg[c[-1]])
            for c in df_agg.columns
            if c[-1] in special_format_agg
        }

        style = df_agg.style.format(
            formatter,
            precision=self.n_decimals,
        )

        # Apply the highlight function to the specified columns
        for i in range(1, k + 1):
            style.apply(
                partial(self._find_k_th_fn, fn=highlight_fn, k=i, props=props[i - 1]),
                subset=([c for c in df_agg.columns if c[-1] == self.main_agg]),
            )

        if self.hide_agg_labels:
            style = style.hide(axis="columns", level=df_agg.columns.nlevels - 1)
        return style

    def save_to_latex(
        self,
        style: Styler,
        filename: str = "table.tex",
        cols_sep: Union[str, int, None] = 0,
        **kwargs,
    ):
        """
        Saves the styled dataframe to a LaTeX file.

        :param style: The styled dataframe to be saved.
        :param filename: The name of the LaTeX file.
        """
        if "convert_css" in kwargs and not kwargs["convert_css"]:
            print(
                "Warning: 'convert_css' has to be set to True for most cases. Setting to True."
            )
            del kwargs["convert_css"]
        if cols_sep is not None:
            if isinstance(cols_sep, int):
                # cols reprents the level of the multindex we want to separate
                assert cols_sep <= style.data.columns.nlevels - 1
                kwargs["column_format"] = "c"
                prev_col_name = "************"
                for c in style.data.columns:
                    if c[cols_sep] != prev_col_name:
                        kwargs["column_format"] += "|c"
                        prev_col_name = c[cols_sep]
                    else:
                        kwargs["column_format"] += "c"
            elif isinstance(cols_sep, str):
                kwargs["column_format"] = cols_sep

        latex = style.to_latex(
            convert_css=True,
            **kwargs,
        )
        with open(filename, "w") as f:
            f.write(latex)
