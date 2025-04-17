import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
from datetime import timedelta
import datetime

import logging

# Logging config
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for verbose logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def _create_lagged_features_predict(data, columnlist, num_lags):
    df = data.copy()
    for col in columnlist:
        for lag in range(1, num_lags + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df.dropna()

def create_multivariate_lagged_features(data, columns, num_lags=12):
    data = data.copy()
    for col in columns:
        for lag in range(1, num_lags + 1):
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)
    return data.dropna()

def train_export_model (
    df: pd.DataFrame,
    columnlist: list,
    PLACE_TO_TEST: int,
    NUM_LAGS: int,
    base_model,
    base_model_name: str, # also used to name model for export, keep short and precise
    train_split_ratio: float = 0.7,
    model_dir: str = 'saved_models'
):
  # raise Exception (f"\nTraining {base_model_name} Model for Place={PLACE_TO_TEST}, base_model = {base_model}")
  # Ensure 'date' is datetime
  df['date'] = pd.to_datetime(df['date'])

  # Filter data
  df_location = df[df['Place'] == PLACE_TO_TEST].copy()
  df_location.set_index('date', inplace=True)

  # Preprocess: interpolate & fill
  for col in columnlist:
    df_location[col] = df_location[col].interpolate(method='linear')

  #Validate train_split_ratio
  if (0.1>train_split_ratio or 1<train_split_ratio): raise Exception(f"Train split ratio is invalid")
  # Generate features
  df_features = create_multivariate_lagged_features(df_location, columnlist, NUM_LAGS)

  # Prepare X and y (multi-output)
  X = df_features.drop(columns=columnlist).select_dtypes(include=[np.number])
  # print (X.columns) X have lag of all features + Place,day,month,year. Cur = 12*9 + 4 = 112 columns
  Y = df_features[columnlist]
  X, Y = X.align(Y, join="inner", axis=0)

  # Split if train_split_ratio < 1
  split_idx = int(len(X) * train_split_ratio)
  X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
  Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]

  # Model
  multi_model = MultiOutputRegressor(base_model)
  multi_model.fit(X_train, Y_train)

  # Save model
  model_path = f"{model_dir}/{base_model_name}_multitarget_place{PLACE_TO_TEST}_{datetime.date.today().strftime('%d%m%y')}.pkl"
  joblib.dump(multi_model, model_path)
  print(f"Multivariate model saved to: {model_path}")

  # Evaluation and Plot
  if (train_split_ratio == 1): eval_dict = None
  else:
    # Predict
    Y_pred = multi_model.predict(X_test)

    # Convert predictions to DataFrame
    prediction_df = pd.DataFrame(Y_pred, columns=[f'predicted_{col}' for col in columnlist], index=Y_test.index)
    for col in columnlist:
        prediction_df[f'actual_{col}'] = Y_test[col]
        eval_dict = {}
        for col in columnlist:
            y_true = Y_test[col]
            y_pred = prediction_df[f'predicted_{col}']
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            eval_dict[col] = {'rmse': rmse, 'mae': mae, 'r2':r2}

  return model_path,eval_dict

def predict_future_steps(
    df: pd.DataFrame,
    freq_days: int,
    place_to_test: int,
    element_column: list,  # e.g. ['pH', 'WQI', 'DO']
    n_steps: int,
    model_path: str,
    num_lags: int = 12,
):
    print(f"\n--- Predicting {n_steps} steps ahead for Place={place_to_test}, Elements={element_column} ---")

    required_cols = ['date', 'Place'] + element_column
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame missing required columns: {required_cols}")
        return None

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    df['date'] = pd.to_datetime(df['date'])
    df_location = df[df['Place'] == place_to_test].copy()
    df_location.set_index('date', inplace=True)

    for col in element_column:
        df_location[col] = df_location[col].interpolate(method='linear')

    df_predict_recursive = df_location.copy()
    current_last_date = df_predict_recursive.index[-1]

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    prediction_output = {col: [] for col in element_column}

    for step in range(n_steps):
        pred_date = current_last_date + timedelta(days=freq_days)
        print(f"\nStep {step+1}/{n_steps} - Predicting for {pred_date.strftime('%Y-%m-%d')}")

        # Create lagged features
        df_with_lags = _create_lagged_features_predict(df_predict_recursive, element_column, num_lags)
        last_features = df_with_lags.drop(columns=element_column).iloc[[-1]]

        if last_features.isnull().values.any():
            last_features.fillna(method='ffill', inplace=True)
            last_features.fillna(method='bfill', inplace=True)

        # Predict and collect results
        # print (last_features.shape)
        # print (last_features)
        predictions = model.predict(last_features)[0]
        # raise Exception (predictions)
        pred_dict = dict(zip(element_column, predictions))

        for col in element_column:
            prediction_output[col].append({
                'predicted_date': pred_date,
                'predicted_value': pred_dict[col]
            })

        # Construct new row with predicted values for all elements
        new_row = pd.DataFrame({
          **{col: [pred_dict[col]] for col in element_column},
          'Place': place_to_test,
          'day': pred_date.day,
          'month': pred_date.month,
          'year': pred_date.year
        }, index=[pred_date])

        # Append new row to recursive df
        df_predict_recursive = pd.concat([df_predict_recursive, new_row])
        # print (df_predict_recursive.shape)
        current_last_date = pred_date

    print("\n--- Multi-step prediction complete ---")
    return prediction_output
