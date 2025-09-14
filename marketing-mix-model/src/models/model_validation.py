from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true: np.array, y_pred: np.array) -> dict:
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    @staticmethod
    def sensitivity_analysis(model, df: pd.DataFrame, feature: str,
                             changes: list = [-0.2, -0.1, 0.1, 0.2]) -> pd.DataFrame:
        results = []
        base_pred = model.predict(df)[1].mean()
        
        for change in changes:
            df_mod = df.copy()
            df_mod[feature] *= (1 + change)
            pred_mod = model.predict(df_mod)[1].mean()
            pct_change = (pred_mod - base_pred) / base_pred * 100
            
            results.append({
                'feature': feature,
                'change': change * 100,
                'revenue_impact_pct': pct_change
            })
        
        return pd.DataFrame(results)
