import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class MarketingMixModel:
    def __init__(self, config):
        self.config = config
        self.mediator_model = Ridge(random_state=config['model']['random_state'])
        self.revenue_model = xgb.XGBRegressor(
            random_state=config['model']['random_state'],
            objective='reg:squarederror'
        )
        self.mediator_scaler = StandardScaler()
        self.revenue_scaler = StandardScaler()
        self.feature_names = None

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        media_cols = self.config['features']['media_channels']
        other_cols = self.config['features']['other_features']
        mediator = self.config['features']['mediator']
        target = self.config['features']['target']

        self.feature_names = media_cols + other_cols

        mediator_features = [col for col in media_cols if col != mediator]
        X_mediator = df[mediator_features + other_cols]
        y_mediator = df[mediator]

        X_revenue = df[self.feature_names]
        y_revenue = df[target]

        return X_mediator, y_mediator, X_revenue, y_revenue

    def fit(self, df: pd.DataFrame):
        X_mediator, y_mediator, X_revenue, y_revenue = self.prepare_features(df)

        X_mediator_scaled = pd.DataFrame(
            self.mediator_scaler.fit_transform(X_mediator),
            columns=X_mediator.columns
        )

        X_revenue_scaled = pd.DataFrame(
            self.revenue_scaler.fit_transform(X_revenue),
            columns=X_revenue.columns
        )

        self.mediator_model.fit(X_mediator_scaled, y_mediator)
        self.revenue_model.fit(X_revenue_scaled, y_revenue)

        return self

    def predict(self, df: pd.DataFrame) -> tuple:
        X_mediator, _, X_revenue, _ = self.prepare_features(df)

        X_mediator_scaled = pd.DataFrame(
            self.mediator_scaler.transform(X_mediator),
            columns=X_mediator.columns
        )

        X_revenue_scaled = pd.DataFrame(
            self.revenue_scaler.transform(X_revenue),
            columns=X_revenue.columns
        )

        mediator_pred = self.mediator_model.predict(X_mediator_scaled)
        revenue_pred = self.revenue_model.predict(X_revenue_scaled)

        return mediator_pred, revenue_pred

    def get_feature_importance(self) -> dict:
        return {
            'mediator_model': dict(zip(self.mediator_model.coef_.index if hasattr(self.mediator_model.coef_, 'index') else [f'feature_{i}' for i in range(len(self.mediator_model.coef_))], self.mediator_model.coef_)),
            'revenue_model': dict(zip(self.feature_names, self.revenue_model.feature_importances_))
        }
