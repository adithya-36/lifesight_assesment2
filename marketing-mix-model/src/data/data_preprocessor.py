import numpy as np
import pandas as pd

class DataPreprocessor:
    def handle_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal features."""
        df = df.copy()
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        return df

    def handle_zero_spend(self, df: pd.DataFrame, media_cols: list) -> pd.DataFrame:
        """Apply log1p transformation to media spend columns to handle zero values."""
        df = df.copy()
        for col in media_cols:
            df[f'{col}_log'] = np.log1p(df[col])
        return df

    def process_data(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Main preprocessing pipeline."""
        df = self.handle_seasonality(df)

        media_cols = config['features']['media_channels']
        df = self.handle_zero_spend(df, media_cols)

        return df
