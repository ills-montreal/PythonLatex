from typing import Union, List, Dict, Optional
from copy import deepcopy

import pandas as pd
import numpy as np

from pythonlatex.highlight_formatter import HighlightFormatter


class PandasTable:
    def __init__(
        self,
        df: pd.DataFrame,
        higlighter: Union[str, HighlightFormatter] = "max",
        n_decimals: int = 3,
        rotate: str = "",
        add_std: bool = True,
        multicols: Optional[Dict[str, List[str]]] = None,
    ):
        self.n_decimals = n_decimals
        self.rotate = rotate
        self.add_std = add_std
        self.multicols = multicols

        self.highlight_method: HighlightFormatter = (
            HighlightFormatter(
                highlight_method="max",
                highlight_format=["\\textbf{"],
            )
            if isinstance(higlighter, str)
            else higlighter
        )

        self.or_df = df.copy()
        self.style: pd.DataFrame = None
        self.latex: str = ""

        self.rows: str = ""
        self.cols: str = ""
        self.values: str = ""

    def _aggregate_results_with_std(self, df_base: pd.DataFrame) -> pd.DataFrame:
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
                    "name": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
                    "category": ["A", "A", "B", "B", "C", "C", "A", "A", "B", "B", "C", "C"],
                    "value": [0,1,2,2,0,1,0,1,2,2,0,1],
                }
            )
            self._aggregate_results_with_std(df_base)

            >> Output:
                                    |       category1                        |       category2
                 avg mean   avg std   A mean    A std    B mean      B std      C mean        C std
            a       1        0.9        0.5       0.7        2          0          0.5        0.7
            b       1        0.9        0.5       0.7        2          0          0.5        0.7


        :param df_base: The base dataframe containing the data to be aggregated.

        :return:  A formatted dataframe containing aggregated mean and standard
        deviation values.
        """

        # Get mean and std of the dataframe
        df_m = df_base.groupby([self.rows, self.cols])[self.values].mean().reset_index()
        df_m[self.cols] = df_m[self.cols] + " mean"
        # Get std of the dataframe
        df_v = df_base.groupby([self.rows, self.cols])[self.values].std().reset_index()
        df_v[self.cols] = df_v[self.cols] + " std"
        # Join the mean and std dataframes to a new one
        df_mean_std = df_m.pivot_table(
            index=self.rows, columns=self.cols, values=self.values
        ).join(df_v.pivot_table(index=self.rows, columns=self.cols, values=self.values))
        df_mean_std.dropna(axis=1, inplace=True)
        df_mean_std.index.name = None

        # Get the average over all values
        full_avg = df_m.pivot_table(
            index=self.rows, columns=self.cols, values=self.values
        ).mean(axis=1)
        ful_std = df_m.pivot_table(
            index=self.rows, columns=self.cols, values=self.values
        ).std(axis=1)

        if self.multicols is not None:
            m = self._get_multicols_with_mean_std(df_mean_std)
            df_mean_std[" mean"] = full_avg
            df_mean_std[" std"] = ful_std
            self._round_values(df_mean_std)
            df_mean_std.columns = pd.MultiIndex.from_tuples(
                [tuple(m[c]) for c in df_mean_std.columns]
            )
        else:
            df_mean_std["avg"] = full_avg
            df_mean_std["avg std"] = ful_std
            self._round_values(df_mean_std)
            df_mean_std.columns = [c.replace(" mean", "") for c in df_mean_std.columns]

        df_mean_std.index = df_mean_std.index.str.replace("_", " ")
        return df_mean_std

    def _get_multicols_with_mean_std(self, df) -> Dict[str, List[str]]:
        """
        From a dictionary of multicols, get the multicols for the new dataframe, where mean and
        std are added to the column names.
        :param df: The dataframe to get the multicols from

        :return: A dictionary with the new column names and the multicols
        """
        if self.multicols is None:
            raise ValueError("Multicols is None")
        m = {}
        n_multicols: int = -1
        for c in df.columns:
            col = c.replace(" mean", "").replace(" std", "")
            assert col in self.multicols, f"Column name {col} not found in multicols"
            m[c] = deepcopy(self.multicols[col])
            m[c][-1] = m[c][-1] + c.split(" ")[1].replace("mean", "").replace(
                "std", " std"
            )
            assert n_multicols == -1 or n_multicols == len(
                m[c]
            ), "Number of multicols is not consistent"
            n_multicols = len(m[c])
        m[" mean"] = [""] * (n_multicols - 1) + ["avg"]
        m[" std"] = [""] * (n_multicols - 1) + ["avg std"]
        return m

    def _round_values(self, df: pd.DataFrame):
        """
        Round the values of the dataframe to the specified number of decimals.
        :param df:
        :return:
        """
        for col in df.columns:
            df[col] = df[col].apply(lambda x: np.round(x, self.n_decimals))

    def style_df_ci(self, df: pd.DataFrame):
        if self.multicols is not None:
            cols_to_keep = [col for col in df.columns if not col[-1].endswith("std")]
        else:
            cols_to_keep = [col for col in df.columns if not col.endswith("std")]
        df = self.highlight_method(df, cols_to_keep, add_std=self.add_std)
        df = df[cols_to_keep]
        if self.rotate == "+":
            col_prefix = "\\rotatebox{90}{\\shortstack{"
        elif self.rotate == "-":
            col_prefix = "\\rotatebox{-90}{\\shortstack{"
        else:
            col_prefix = ""

        def join_col_names(col: Union[List[str], str]) -> str:
            if isinstance(col, str):
                return col
            c = ""
            for i, name in enumerate(col):
                if i > 0:
                    c += "\\\\"
                c += name
            return c

        df.columns = [
            col_prefix
            + join_col_names(col)
            + "}" * (col_prefix.count("{") - col_prefix.count("}"))
            for col in df.columns
        ]

        style = df.style
        col_format = "r|"
        prev_cols = "This is not a column name that will be used"
        for col in style.columns:
            ov_col = col[len(col_prefix) :].split(" \\\\")[0]
            if prev_cols != ov_col:
                col_format += "|"
                prev_cols = ov_col
            col_format += "c"
        col_format += "|"

        latex = style.to_latex(
            column_format=col_format,
            siunitx=True,
        )
        return df, style, latex

    def __getitem__(self, item):
        """
        Returns the processed dataframe, where rows are item[0] and columns are item[1] and values are item[2]
        :param item:
            item[0]: rows
            item[1]: columns
            item[2]: values
        :return:
        """
        self.rows, self.cols, self.values = item
        df = self.or_df.copy()
        df = self._aggregate_results_with_std(df)
        df, style, latex = self.style_df_ci(df)

        self.style = style
        self.latex = latex

        return df

    def save_latex(self, path: str):
        """
        Save the latex table to a file
        :param path:
            Path to save the latex table
        :return:
        """
        if self.latex is None:
            raise ValueError(
                "Latex table not generated yet. Use __getitem__ to generate it."
            )

        with open(path, "w") as f:
            f.write(self.latex)
