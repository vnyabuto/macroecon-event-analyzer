print("[model_xgboost.py] v1.0 loaded")

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

def predict_sector_movements(df: pd.DataFrame, sector: str):
    """
    Trains an XGBoost classifier to predict up/down movement of `sector` based on macro features.
    Returns (accuracy, predictions_series, model), or (None, None, None) if something goes wrong.
    """
    if sector not in df.columns or df.shape[0] < 10:
        print(f"[XGBOOST] Insufficient data or missing sector: {sector}")
        return None, None, None

    try:
        data = df.copy()
        data['Target'] = (data[sector].pct_change().shift(-1) > 0).astype(int)

        # Drop rows with NaNs
        data = data.dropna()
        if data.empty:
            print(f"[XGBOOST] Data is empty after dropna()")
            return None, None, None

        X = data.drop(columns=[sector, 'Target'])
        y = data['Target']

        # Ensure all features are numeric
        if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
            print("[XGBOOST] Non-numeric feature types found:")
            print(X.dtypes)
            return None, None, None

        # Log preview
        print(f"[XGBOOST] X shape: {X.shape}, y shape: {y.shape}")
        print(f"[XGBOOST] Feature preview:\n{X.head()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        preds_series = pd.Series(preds, index=y_test.index)

        return accuracy, preds_series, model

    except Exception as e:
        print(f"[XGBOOST] prediction failed for sector={sector}: {e}")
        return None, None, None



def get_feature_importance(model, feature_names):
    """
    Returns a Plotly bar chart of feature importances from the trained XGBoost model.
    If model is None, returns an empty placeholder chart.
    """
    if model is None:
        return px.bar(title="No model available for feature importance")

    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    fig = px.bar(
        df_imp,
        x='Feature',
        y='Importance',
        title='Feature Importance (XGBoost)'
    )
    return fig
