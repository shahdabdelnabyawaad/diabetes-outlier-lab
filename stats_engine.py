import numpy as np
import pandas as pd


class StatisticalDetector:
    """
    Statistical Detector for detecting and treating outliers in numerical data.

    This class provides methods for:
    - Detecting outliers using IQR, Z-Score.
    - Calculating IQR fences.
    - Treating outliers using Trimming, Winsorization, Imputation.
    - Flagging "zero as missing" values (common in clinical datasets such as
      the diabetes dataset, where 0 is not a physiologically valid reading
      for columns like Glucose or BMI).
    - Generating statistical summaries.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the StatisticalDetector.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        None
        """

        self.df = df.copy()


    def calculate_iqr_fences(
        self,
        column: str
    ) -> dict[str, float]:
        """
        Calculate Q1, Q3, IQR, and the lower and upper fences.

        Parameters
        ----------
        column : str
            Name of the numerical column.

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - Q1
            - Q3
            - IQR
            - lower_fence
            - upper_fence
        """

        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR

        return {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence
        }


    def detect_outliers_iqr(
        self,
        column: str
    ) -> dict[str, object]:
        """
        Detect outliers in a numerical column using the IQR method.

        Parameters
        ----------
        column : str
            Name of the numerical column.

        Returns
        -------
        dict[str, object]
            Dictionary containing:
            - outlier_mask : pd.Series
                Boolean mask identifying outliers.
            - outliers : pd.DataFrame
            - count : int
            - Q1 : float
            - Q3 : float
            - IQR : float
            - lower_fence : float
            - upper_fence : float
        """

        fences = self.calculate_iqr_fences(column)

        mask = (
            (self.df[column] < fences["lower_fence"])
            | (self.df[column] > fences["upper_fence"])
        )

        outliers = self.df[mask]

        return {
            "outlier_mask": mask,
            "outliers": outliers,
            "count": len(outliers),
            "Q1": fences["Q1"],
            "Q3": fences["Q3"],
            "IQR": fences["IQR"],
            "lower_fence": fences["lower_fence"],
            "upper_fence": fences["upper_fence"]
        }


    def detect_outliers_zscore(
        self,
        column: str,
        threshold: float = 3.0
    ) -> dict[str, object]:
        """
        Detect outliers using the Z-Score method.

        Parameters
        ----------
        column : str
            Name of the numerical column.
        threshold : float, default=3.0
            Z-Score threshold used to identify outliers.

        Returns
        -------
        dict[str, object]
            Dictionary containing:
            - outlier_mask : pd.Series
            - outliers : pd.DataFrame
            - count : int
            - mean : float
            - std : float
            - threshold : float
            - z_scores : pd.Series
        """

        mean = self.df[column].mean()
        std = self.df[column].std()

        z_scores = (self.df[column] - mean) / std

        mask = np.abs(z_scores) > threshold

        outliers = self.df[mask]

        return {
            "outlier_mask": mask,
            "outliers": outliers,
            "count": len(outliers),
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "z_scores": z_scores
        }


    def detect_zero_as_missing(
        self,
        column: str
    ) -> dict[str, object]:
        """
        Flag zero values in a column where zero is not a physiologically
        valid measurement (e.g. Glucose, BloodPressure, SkinThickness,
        Insulin, BMI in the diabetes dataset). These are almost always
        undocumented missing values rather than genuine readings.

        Parameters
        ----------
        column : str
            Name of the numerical column.

        Returns
        -------
        dict[str, object]
            Dictionary containing:
            - zero_mask : pd.Series
            - count : int
            - percentage : float
        """

        mask = self.df[column] == 0
        count = int(mask.sum())

        return {
            "zero_mask": mask,
            "count": count,
            "percentage": (count / len(self.df)) * 100 if len(self.df) else 0.0
        }


    def trim_outliers(
        self,
        column: str,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove rows containing outliers from the dataset.

        Parameters
        ----------
        column : str
            Name of the numerical column.
        method : str, default="iqr"
            Outlier detection method.
            Accepted values:
            - "iqr"
            - "zscore"
        threshold : float, default=3.0
            Z-Score threshold. Used only when method="zscore".

        Returns
        -------
        pd.DataFrame
            A new DataFrame with outlier rows removed.

        Raises
        ------
        ValueError
            If an unsupported detection method is provided.
        """

        if method.lower() == "iqr":
            result = self.detect_outliers_iqr(column)

        elif method.lower() == "zscore":
            result = self.detect_outliers_zscore(
                column,
                threshold
            )

        else:
            raise ValueError(
                "Method must be 'iqr' or 'zscore'."
            )

        mask = result["outlier_mask"]

        cleaned_df = self.df[~mask].copy()

        return cleaned_df


    def winsorize_outliers(
        self,
        column: str,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Replace outlier values with predefined boundary values.

        For IQR:
        Values below the lower fence or above the upper fence
        are replaced by the lower fence or upper fence.

        For Z-Score:
        Values below mean - threshold * std or above mean + threshold * std
        are replaced by that limit.

        Parameters
        ----------
        column : str
            Name of the numerical column.
        method : str, default="iqr"
            Outlier detection method.
            Accepted values:
            - "iqr"
            - "zscore"
        threshold : float, default=3.0
            Z-Score threshold. Used only when method="zscore".

        Returns
        -------
        pd.DataFrame
            A new DataFrame with outlier values capped at the
            calculated boundaries.

        Raises
        ------
        ValueError
            If an unsupported detection method is provided.
        """

        cleaned_df = self.df.copy()

        if method.lower() == "iqr":

            result = self.calculate_iqr_fences(column)

            cleaned_df[column] = cleaned_df[column].clip(
                lower=result["lower_fence"],
                upper=result["upper_fence"]
            )

        elif method.lower() == "zscore":

            result = self.detect_outliers_zscore(
                column,
                threshold
            )

            mean = result["mean"]
            std = result["std"]

            lower_limit = mean - threshold * std
            upper_limit = mean + threshold * std

            cleaned_df[column] = cleaned_df[column].clip(
                lower=lower_limit,
                upper=upper_limit
            )

        else:
            raise ValueError(
                "Method must be 'iqr' or 'zscore'."
            )

        return cleaned_df


    def impute_outliers(
        self,
        column: str,
        method: str = "iqr",
        strategy: str = "median",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Replace detected outliers with a statistical value.

        Supported replacement strategies:
        - Mean
        - Median

        Parameters
        ----------
        column : str
            Name of the numerical column.
        method : str, default="iqr"
            Outlier detection method.
            Accepted values:
            - "iqr"
            - "zscore"
        strategy : str, default="median"
            Replacement strategy.
            Accepted values:
            - "mean"
            - "median"
        threshold : float, default=3.0
            Z-Score threshold. Used only when method="zscore".

        Returns
        -------
        pd.DataFrame
            A new DataFrame with outliers replaced by the
            selected statistical value.

        Raises
        ------
        ValueError
            If an unsupported detection method or strategy
            is provided.
        """

        cleaned_df = self.df.copy()

        if method.lower() == "iqr":

            result = self.detect_outliers_iqr(column)

        elif method.lower() == "zscore":

            result = self.detect_outliers_zscore(
                column,
                threshold
            )

        else:
            raise ValueError(
                "Method must be 'iqr' or 'zscore'."
            )

        mask = result["outlier_mask"]

        if strategy.lower() == "median":

            replacement = self.df[column].median()

        elif strategy.lower() == "mean":

            replacement = self.df[column].mean()

        else:
            raise ValueError(
                "Strategy must be 'mean' or 'median'."
            )

        cleaned_df.loc[mask, column] = replacement

        return cleaned_df


    def statistical_summary(
        self,
        column: str
    ) -> dict[str, float]:
        """
        Generate a statistical summary for a numerical column.

        Parameters
        ----------
        column : str
            Name of the numerical column.

        Returns
        -------
        dict[str, float]
            Dictionary containing the statistical measures.
        """

        series = self.df[column]

        return {
            "count": float(series.count()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "Q1": float(series.quantile(0.25)),
            "Q3": float(series.quantile(0.75)),
            "max": float(series.max())
        }
