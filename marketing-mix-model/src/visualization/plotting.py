import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class MMMVisualizer:
    def __init__(self, style='darkgrid'):
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = [12, 6]
        
    def plot_spend_over_time(self, df: pd.DataFrame, media_channels: list):
        """Plot media spend trends over time"""
        fig, ax = plt.subplots()
        for channel in media_channels:
            ax.plot(df['date'], df[channel], label=channel, alpha=0.7)
        ax.set_title('Media Spend Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Spend')
        ax.legend()
        plt.xticks(rotation=45)
        return fig
    
    def plot_feature_importance(self, importance_dict: dict):
        """Plot feature importance for both models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mediator model importance
        med_imp = pd.Series(importance_dict['mediator_model']).sort_values()
        med_imp.plot(kind='barh', ax=ax1)
        ax1.set_title('Mediator Model\nFeature Importance')
        
        # Revenue model importance
        rev_imp = pd.Series(importance_dict['revenue_model']).sort_values()
        rev_imp.plot(kind='barh', ax=ax2)
        ax2.set_title('Revenue Model\nFeature Importance')
        
        plt.tight_layout()
        return fig
    
    def plot_model_diagnostics(self, y_true: np.array, y_pred: np.array, title: str):
        """Plot model diagnostics including actual vs predicted and residuals"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'{title}\nActual vs Predicted')
        
        # Residuals
        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_title(f'{title}\nResidual Distribution')
        ax2.set_xlabel('Residual Value')
        
        plt.tight_layout()
        return fig