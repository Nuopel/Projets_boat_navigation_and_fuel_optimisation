"""Data loading module for UCI Yacht Hydrodynamics dataset.

This module provides the YachtDataLoader class for loading and validating
the UCI Yacht Hydrodynamics dataset from Excel format.
"""

from typing import List
import pandas as pd
import numpy as np


class YachtDataLoader:
    """Load and validate UCI Yacht Hydrodynamics dataset.

    The UCI Yacht Hydrodynamics dataset contains 308 experimental measurements
    from 22 yacht hull designs tested at the Delft Ship Hydromechanics Laboratory.

    Attributes:
        filepath: Path to the yacht_hydro.xls file
        data: Loaded DataFrame (None until load() is called)

    Example:
        >>> loader = YachtDataLoader('data/yacht_hydro.xls')
        >>> df = loader.load()
        >>> print(df.shape)
        (308, 7)
    """

    # Expected column names from UCI dataset
    EXPECTED_COLUMNS = [
        'Longitudinal_position',  # Longitudinal position of center of buoyancy
        'Prismatic_coefficient',  # Prismatic coefficient
        'Length_displacement_ratio',  # Length-displacement ratio
        'Beam_draught_ratio',  # Beam-draught ratio
        'Length_beam_ratio',  # Length-beam ratio
        'Froude_number',  # Froude number
        'Residuary_resistance'  # Residuary resistance per unit weight of displacement
    ]

    EXPECTED_SHAPE = (308, 7)

    def __init__(self, filepath: str):
        """Initialize the data loader.

        Args:
            filepath: Path to the yacht_hydro.xls Excel file
        """
        self.filepath = filepath
        self.data = None

    def load(self) -> pd.DataFrame:
        """Load the yacht hydrodynamics dataset from Excel file.

        Returns:
            DataFrame with 308 rows and 7 columns containing yacht measurements

        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the loaded data doesn't match expected format
        """
        try:
            # UCI yacht_hydro.xls is actually a CSV file despite the extension
            # Load as CSV with the first row as header
            self.data = pd.read_csv(self.filepath)

            # Rename columns to our standardized names
            column_mapping = {
                'LC': 'Longitudinal_position',
                'PC': 'Prismatic_coefficient',
                'L/D': 'Length_displacement_ratio',
                'B/Dr': 'Beam_draught_ratio',
                'L/B': 'Length_beam_ratio',
                'Fr': 'Froude_number',
                'Rr': 'Residuary_resistance'
            }
            self.data = self.data.rename(columns=column_mapping)

            # Validate the loaded data
            if not self.validate():
                raise ValueError(
                    f"Data validation failed. Expected shape {self.EXPECTED_SHAPE}, "
                    f"got {self.data.shape}"
                )

            return self.data

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Yacht hydrodynamics data file not found: {self.filepath}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Error loading yacht data from {self.filepath}: {str(e)}"
            ) from e

    def validate(self) -> bool:
        """Validate that loaded data matches expected format.

        Checks:
        - Correct shape (308 rows, 7 columns)
        - No missing values
        - All resistance values are positive
        - All numeric data types

        Returns:
            True if all validation checks pass, False otherwise
        """
        if self.data is None:
            return False

        # Check shape
        if self.data.shape != self.EXPECTED_SHAPE:
            return False

        # Check for missing values
        if self.data.isnull().any().any():
            return False

        # Check all columns are numeric
        if not all(self.data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            return False

        # Check resistance values are positive
        if not (self.data['Residuary_resistance'] > 0).all():
            return False

        return True

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names (excluding target).

        Returns:
            List of 6 feature column names
        """
        return self.EXPECTED_COLUMNS[:-1]

    def get_target_name(self) -> str:
        """Get name of the target column.

        Returns:
            Name of the resistance target column
        """
        return self.EXPECTED_COLUMNS[-1]

    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics of the loaded dataset.

        Returns:
            DataFrame with statistical summary (mean, std, min, max, etc.)

        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.data is None:
            raise ValueError("Data must be loaded before getting summary statistics")

        return self.data.describe()
