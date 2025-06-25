import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

def predict_sector_movements(df, sector_name):
    """
    Trains an XGBoost classifier to predict sector price direction (up/down) based on macro features.

    Args:
        df (pd.DataFrame): DataFrame with macroeconomic features and sector prices.
        sector_name (str): Name of the sector column to predict movement for.

    Returns:
        accuracy (float): Accuracy of the model.
        predictions (pd.Series): Predicted direction (0 = down, 1 = up).
        model (xgb.XGBClassifier): Trained model.
    """
    df = df.dropna()
    df['Target'] = (df[sector_name].diff().shift(-1) > 0).astype(int)

    features = df.drop(columns=[sector_name, 'Target'])
    target = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, shuffle=False)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy, pd.Series(predictions, index=X_test.index), model

def get_feature_importance(model, feature_names):
    """
    Generates a bar plot of feature importances from a trained XGBoost model.

    Args:
        model (xgb.XGBClassifier): Trained XGBoost model.
        feature_names (list): List of feature names.

    Returns:
        fig (plotly.graph_objects.Figure): Feature importance plot.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
    return fig

print("DEBUG: model_xgboost.py loaded")

def predict_sector_movements(df, sector_name):
    print("DEBUG: predict_sector_movements called")
    ...