import pandas as pd
import yaml
from pathlib import Path

class DataLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw marketing mix data from CSV file."""
        data_path = Path(self.config['data']['raw_path'])
        df = pd.read_csv(data_path)
        
        # Convert 'week' to datetime and drop original column
        if 'week' in df.columns:
            df['date'] = pd.to_datetime(df['week'])
            df = df.drop(columns=['week'])
        
        return df
    
    def get_feature_columns(self) -> dict:
        """Get feature column names from config."""
        return {
            'media_channels': self.config['features']['media_channels'],
            'other_features': self.config['features']['other_features'],
            'target': self.config['features']['target'],
            'mediator': self.config['features']['mediator']
        }
