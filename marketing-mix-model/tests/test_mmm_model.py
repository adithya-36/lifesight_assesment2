import pytest
import numpy as np
import pandas as pd
from src.models.mmm_model import MarketingMixModel
from src.models.model_validation import ModelEvaluator

def test_model_fitting():
    data = {
        'facebook_spend': np.random.rand(100),
        'google_spend': np.random.rand(100),
        'tiktok_spend': np.random.rand(100),
        'snapchat_spend': np.random.rand(100),
        'instagram_spend': np.random.rand(100),
        'social_followers': np.random.rand(100),
        'average_price': np.random.rand(100),
        'promotions': np.random.rand(100),
        'emails_send': np.random.rand(100),
        'sms_send': np.random.rand(100),
        'revenue': np.random.rand(100)
    }
    df = pd.DataFrame(data)

    config = {
        'model': {'random_state': 42},
        'features': {
            'media_channels': ['facebook_spend', 'tiktok_spend', 'google_spend', 'snapchat_spend', 'instagram_spend'],
            'other_features': ['social_followers', 'average_price', 'promotions', 'emails_send', 'sms_send'],
            'target': 'revenue',
            'mediator': 'google_spend'
        }
    }

    model = MarketingMixModel(config)
    model.fit(df)

    mediator_pred, revenue_pred = model.predict(df)
    assert len(mediator_pred) == len(df)
    assert len(revenue_pred) == len(df)

def test_model_evaluation():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 5.1])

    metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert 'mape' in metrics
