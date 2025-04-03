from typing import Union, List, Optional, Any, Callable

import pandas as pd


class HighlightFormatter:
    def __init__(
        self,
        highlight_method: Union[str, Callable[[pd.DataFrame], pd.Series]] = "max",
        highlight_format: List[str] = ["\\textbf{\\underline{", "\\textbf{"],
    ):
        self.highlight_format = highlight_format
        self.highlight_method = highlight_method

    def _get_vals_to_highlight(self, df: pd.DataFrame) -> pd.Series:
        if self.highlight_method == "max":
            return df.max(axis=0)
        elif self.highlight_method == "min":
            return df.min(axis=0)
        elif callable(self.highlight_method):
            out = self.highlight_method(df)
            assert isinstance(
                out, pd.Series
            ), "Highlight method must return a pandas Series"
            return out
        else:
            raise ValueError("Highlight method not recognized")

    def _get_all_masks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        masks = []
        to_keep_for_next_mask = None

        for fmt in self.highlight_format:
            if to_keep_for_next_mask is None:
                vals = self._get_vals_to_highlight(df)
            else:
                vals = self._get_vals_to_highlight(df.where(to_keep_for_next_mask))
            masks.append(df == vals)
            if to_keep_for_next_mask is not None:
                to_keep_for_next_mask = to_keep_for_next_mask & ~masks[-1]
            else:
                to_keep_for_next_mask = ~masks[-1]

        return masks

    def __call__(
        self,
        df: pd.DataFrame,
        cols_to_keep: Optional[List[Any]] = None,
        add_std: bool = True,
    ) -> pd.DataFrame:
        style = df.copy()
        if cols_to_keep is None:
            cols_to_keep = df.columns.unique().tolist()
        masks = self._get_all_masks(df[cols_to_keep])
        for col in style.columns:
            if style.columns.nlevels > 1:
                if not col[-1].endswith("std"):
                    col_std_l = list(col)
                    col_std_l[-1] += " std"
                    col_std = tuple(col_std_l)
                else:
                    continue
            else:
                if not col.endswith("std"):
                    col_std = col + " std"
                else:
                    continue

            if add_std and col_std in style.columns:
                style[col] = (
                    style[col].apply(lambda x: f"{x}")
                    + "$\\pm$ \\tiny "
                    + style[col_std].apply(lambda x: f"{x}")
                )
            else:
                style[col] = style[col].apply(lambda x: f"{x}")

        style = style[cols_to_keep]

        for col in style.columns:
            for i, mask in enumerate(masks):
                for best in mask[mask[col]].index:
                    style.loc[best, col] = (
                        self.highlight_format[i]
                        + style.loc[best, col]
                        + "}" * self.highlight_format[i].count("{")
                    )

        return style
