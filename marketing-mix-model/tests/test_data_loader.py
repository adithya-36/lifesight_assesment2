import pytest
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor

def test_data_loading():
    loader = DataLoader()
    df = loader.load_raw_data()
    assert not df.empty, "Data should not be empty"
    assert 'week' in df.columns, "Week column should be present"

def test_preprocessing():
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    df = loader.load_raw_data()
    features = loader.get_feature_columns()
    
    processed_df = preprocessor.process_data(df, loader.config)
    assert 'week_of_year' in processed_df.columns, "Seasonal features should be added"