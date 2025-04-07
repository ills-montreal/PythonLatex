from pythonlatex.pandas_to_latex import PandasTableFormatter


class TableMeanStd(PandasTableFormatter):
    """
    TableMeanStd is a class that formats a Pandas DataFrame into a LaTeX table with mean and standard deviation.
    It inherits from the PandasTableFormatter class.
    """

    def __init__(
        self,
        n_decimals: int = 3,
    ):
        super().__init__(
            n_decimals=n_decimals,
            aggregation_methods=["mean", "std"],
            main_subset=0,
        )
