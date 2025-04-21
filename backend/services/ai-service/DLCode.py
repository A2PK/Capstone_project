import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
from datetime import timedelta
import datetime
from sklearn.preprocessing import StandardScaler

# --- Config and Loss ---
class Config:
    def __init__(self):
        self.seq_len = 24
        self.label_len = 12 # Not directly used by this ETSformer version
        self.pred_len = 12
        self.enc_in = 7  # NH4N, CODMn, pH, DO + day, month, year
        self.c_out = 4    # NH4N, CODMn, pH, DO (Features to forecast)
        self.d_model = 512
        self.n_heads = 8  # Internal heads within layers
        self.e_layers = 3  # Parallel Encoder Heads
        self.d_layers = 3  # Parallel Decoder Layers (must match e_layers)
        self.d_ff = 2048
        self.K = 1        # Top-k frequencies
        self.dropout = 0.1
        self.activation = 'gelu'
        self.output_attention = False
        self.std = 0.05   # Std dev for Transform
        self.layer_norm_eps=1e-5

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    # Loop needs to stop early enough to get a full prediction sequence y
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_len])
    if not X: # Handle cases where data is too short
        return np.array([]).reshape(0, seq_len, data.shape[1]), \
               np.array([]).reshape(0, pred_len, data.shape[1])
    return np.array(X), np.array(y)

def train_export_model_DL_SingleLocation(
    df: pd.DataFrame,
    elements_list: list,
    PLACE_TO_TEST,
    place_column_name,
    config,
    base_model_class,
    base_model_name: str,
    date_tag,
    train_split_ratio: float = 0.7,
    model_dir: str = 'saved_models'
):
    results = {}
    seq_length = config.seq_len
    pred_length = config.pred_len
    device = 'cpu'
    epochs = config.epochs

    # Filter data for specific location
    try:
        df_location = df[df[place_column_name] == PLACE_TO_TEST].copy()
        df_location.set_index('date', inplace=True)
    except KeyError as e:
        raise KeyError(f"Column not found: {e}. Available columns: {df.columns.tolist()}")

    time_features = ['day', 'month', 'year']
    input_features = elements_list + time_features

    # Initialize StandardScaler
    scaler = StandardScaler()

    train_data_list = []
    test_data_dict = {}

    for place, group in df.groupby(place_column_name):
        group = group.copy()
        for col in input_features:
            if col not in ['date', place_column_name]:
                group[col] = pd.to_numeric(group[col], errors='coerce')

        group = group.dropna(subset=input_features).sort_values(by='date')

        if group.shape[0] < seq_length + pred_length + 10:
            print(f"Skipping {place} due to insufficient data ({group.shape[0]} rows) after dropna.")
            continue

        split_idx = int(len(group) * train_split_ratio)
        train_df = group.iloc[:split_idx].copy()
        test_df = group.iloc[split_idx:].copy()

        train_data_list.append(train_df)
        test_data_dict[place] = test_df

    if not train_data_list:
        print("No locations with sufficient data for training.")
        return None

    train_data_combined = pd.concat(train_data_list)
    X_train_raw = train_data_combined[input_features].values

    # Fit and transform training data with scaler
    X_train_scaled = scaler.fit_transform(X_train_raw)

    # Save the scaler
    scaler_path = os.path.join(model_dir, f"scale_{base_model_name}_multitarget_place{PLACE_TO_TEST}_{date_tag}.pkl")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Create sequences
    X_train_seq, y_train_seq_all_features = create_sequences(X_train_scaled, seq_length, pred_length)

    try:
        feature_indices = [input_features.index(f) for f in elements_list]
    except ValueError as e:
        print(f"Error finding feature index: {e}")
        return None

    y_train_seq = y_train_seq_all_features

    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.FloatTensor(y_train_seq).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = base_model_class(config).float().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_X, None, None, None)
            loss = criterion(outputs, batch_y)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Epoch {epoch+1}: Invalid loss detected: {loss.item()}. Stopping training.")
                return results

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

    # Save model
    model_path = f"{model_dir}/{base_model_name}_multitarget_place{PLACE_TO_TEST}_{date_tag}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # --- Inference ---
    print("\nStarting evaluation...")
    model.eval()

    eval_dict = {}
    for place, test_data in test_data_dict.items():
        if place != PLACE_TO_TEST: continue
        print(f"\nEvaluating {place}...")

        X_test_raw = test_data[input_features].values
        X_test_scaled = scaler.transform(X_test_raw)  # apply same scaling

        test_dates = test_data['date'].values

        try:
            train_df_place = next(df for df in train_data_list if df[place_column_name].iloc[0] == place)
        except (StopIteration, IndexError):
            print(f"Training data missing or invalid for {place}. Skipping.")
            continue

        if len(train_df_place) < seq_length:
            print(f"Skipping evaluation for {place}: Training data too short.")
            continue

        initial_input_raw = train_df_place[input_features].values[-seq_length:]
        initial_input_scaled = scaler.transform(initial_input_raw)

        y_pred_list = []

        with torch.no_grad():
            current_input_seq = initial_input_scaled.copy()

            for i in range(len(X_test_scaled)):
                seq_tensor = torch.FloatTensor(current_input_seq).unsqueeze(0).to(device)
                prediction = model(seq_tensor, None, None, None)
                first_step_pred = prediction[0, 0, :].cpu().numpy()
                y_pred_list.append(first_step_pred)

                if i < len(X_test_scaled):
                    next_step_input = X_test_scaled[i, :]
                else:
                    break

                current_input_seq = np.vstack((current_input_seq[1:], next_step_input))

        y_pred = np.array(y_pred_list)
        num_predictions = len(y_pred)

        if num_predictions == 0:
            print(f"No predictions generated for {place}.")
            continue

        y_test = X_test_scaled[:num_predictions, feature_indices]
        aligned_test_dates = test_dates[:num_predictions]

        print(f"{place} Metrics:")
        metrics_summary = {}
        for i, feature in enumerate(elements_list):
            if len(y_test[:, i]) != len(y_pred[:, i]):
                print(f"Warning: Length mismatch for feature '{feature}' in {place}. Skipping.")
                continue

            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            max_se = np.max(np.sqrt((y_test[:, i] - y_pred[:, i])**2)) if len(y_pred[:, i]) else 0

            metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "HRSE": max_se}
            eval_dict[feature] = metrics
            print(f"  {feature}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}, HRSE={max_se:.3f}")

    return model_path, eval_dict

class HuberLogCoshLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        return torch.mean(torch.where(
            torch.abs(error) < self.delta,
            (error ** 2) / 2,  # MSE-like behavior for small errors
            torch.log(torch.cosh(error))  # Log-Cosh behavior for large errors
        ))

def predict_future_steps_DL(
    df: pd.DataFrame,
    freq_days: int,
    place_to_test,
    place_column_name,
    element_column: list,
    n_steps: int,
    model_path: str,
    scaler_path: str,  # 🔹 Add this
    model_output_length: int,
    num_lags: int = 12,
):
    device = 'cpu'

    print(f"\n--- Predicting {n_steps} steps ahead for Place={place_to_test}, Elements={element_column} ---")

    required_cols = ['date'] + element_column
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame missing required columns: {required_cols}")
        return None

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Model or scaler file not found at {model_path} / {scaler_path}")
        return None

    # Load model and scaler
    try:
        model = joblib.load(model_path)
        scaler: StandardScaler = joblib.load(scaler_path)  # 🔹 Load scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None

    # Extract and preprocess data
    df_location = df[df[place_column_name] == place_to_test].copy()
    df_location['date'] = pd.to_datetime(df_location['date'])
    df_location.set_index('date', inplace=True)

    for col in element_column:
        df_location[col] = df_location[col].interpolate(method='linear')

    df_predict_recursive = df_location.copy()
    current_last_date = df_predict_recursive.index[-1]

    # Prepare input: last num_lags points
    recent_dates = df_predict_recursive.index[-num_lags:]
    recent_data = df_predict_recursive[element_column].values[-num_lags:]

    # Append time features to each row before scaling
    recent_data_with_time = []
    for i in range(num_lags):
        date = recent_dates[i]
        time_feats = [date.day, date.month, date.year]
        full_row = np.concatenate([recent_data[i], time_feats])
        recent_data_with_time.append(full_row)

    recent_data_with_time = np.array(recent_data_with_time)
    recent_data_scaled = scaler.transform(recent_data_with_time)

    recent_dates = df_predict_recursive.index[-num_lags:]

    initial_input = recent_data_scaled

    current_input_seq = np.vstack(initial_input)

    y_pred = []
    prediction_output = {col: [] for col in element_column}

    with torch.no_grad():
        for step in range(0, n_steps, model_output_length):
            seq_tensor = torch.FloatTensor(current_input_seq).unsqueeze(0).to(device)
            prediction = model(seq_tensor, None, None, None).cpu().numpy()

            current_pred_len = min(model_output_length, n_steps - len(y_pred))
            pred_chunk = prediction[0, :current_pred_len]

            # 🔹 Inverse transform predictions
            pred_chunk_inv = scaler.inverse_transform(pred_chunk)  # only for target columns

            y_pred.extend(pred_chunk_inv)

            for j in range(current_pred_len):
                current_last_date += timedelta(days=freq_days)
                time_feats = [current_last_date.day, current_last_date.month, current_last_date.year]
    
                pred_values = pred_chunk_inv[j].copy()
                pred_values[-3:] = time_feats

                for i, col in enumerate(element_column):
                    prediction_output[col].append({
                        'predicted_date': current_last_date,
                        'predicted_value': pred_values[i]
                    })

                # 🔁 For the next input:
                # - Scale only the target values (exclude time)
                next_input_scaled = scaler.transform([pred_values])  # scale only original vars
                current_input_seq = np.vstack((current_input_seq[1:], next_input_scaled))

    print("\n--- Multi-step prediction complete ---")
    return prediction_output