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
    #load config
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
    input_features = elements_list + time_features # All features for encoder input
    
    # --- Data Splitting and Preprocessing (No Scaling) ---
    train_data_list = []
    test_data_dict = {}

    for place, group in df.groupby('Place'):
        group = group.copy() # Work on a copy to avoid SettingWithCopyWarning
        # Convert features to numeric, coercing errors (replace NaNs later if needed)
        for col in input_features:
             if col not in ['date', 'Place']: # Keep date/place as is for now
                 group[col] = pd.to_numeric(group[col], errors='coerce')

        group = group.dropna(subset=input_features).sort_values(by='date')

        if group.shape[0] < seq_length + pred_length + 10: # Add buffer
            print(f"Skipping {place} due to insufficient data ({group.shape[0]} rows) after dropna.")
            continue

        split_idx = int(len(group) * train_split_ratio)
        train_df = group.iloc[:split_idx].copy()
        test_df = group.iloc[split_idx:].copy()

        # --- No Scaling Applied ---
        train_data_list.append(train_df)
        test_data_dict[place] = test_df

    if not train_data_list:
        print("No locations with sufficient data for training.")
        return None

    # Combine all training data (unscaled)
    train_data_combined = pd.concat(train_data_list)
    X_train_raw = train_data_combined[input_features].values # Use raw values

    X_train_seq, y_train_seq_all_features = create_sequences(X_train_raw, seq_length, pred_length) # Assuming create_sequences is defined/imported

    # Extract only the target feature columns for y
    try:
        feature_indices = [input_features.index(f) for f in elements_list]
    except ValueError as e:
        print(f"Error finding feature index: {e}")
        print(f"Input features: {input_features}")
        print(f"Original features: {elements_list}")
        return None
    y_train_seq = y_train_seq_all_features # [:, :, feature_indices]


    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.FloatTensor(y_train_seq).to(device)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # Adjust batch size based on memory
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # Smaller batch size
    
    #get model object
    model = base_model_class(config).float().to(device)
    # results = train_etsformer_multivariate(df, elements_list, seq_length=4, pred_length=4, epochs=20)
    
    # Loss and optimizer
    criterion = nn.MSELoss() # Using MSE for simplicity
    # Consider adjusting learning rate, maybe lower if data has large values
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.float().to(device) # Ensure float type and on correct device
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_X, None, None, None)

            # Loss expects (b, pred_len, c_out) vs (b, pred_len, c_out)
            loss = criterion(outputs, batch_y)

            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Epoch {epoch+1}: Invalid loss detected: {loss.item()}. Stopping training.")
                return results 

            loss.backward()
            # Gradient clipping is important, especially without scaling
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss) # Step scheduler based on average epoch loss

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

    # --- Save the model ---
    model_path = f"{model_dir}/{base_model_name}_multitarget_place{PLACE_TO_TEST}_{date_tag}.pkl"
    joblib.dump(model, model_path)
    print(f"Multivariate model saved to: {model_path}")
    
    # --- Inference and Evaluation ---
    print("\nStarting evaluation...")
    model.eval()
    
    eval_dict = {}
    for place, test_data in test_data_dict.items():
        if place != PLACE_TO_TEST: continue
        print(f"\nEvaluating {place}...")
        X_test = test_data[input_features].values # Use raw test data
        test_dates = test_data['date'].values # Dates for potential later use (though not plotting)

        # Find the corresponding train_df for this place
        try:
            train_df_place = next(df for df in train_data_list if df['Place'].iloc[0] == place)
        except StopIteration:
            print(f"Could not find training data for place {place} during evaluation. Skipping.")
            continue
        except IndexError:
             print(f"Training data for place {place} seems empty. Skipping.")
             continue


        # Get the last sequence from this place's training data (raw)
        if len(train_df_place) < seq_length:
             print(f"Skipping evaluation for {place}: Training data too short ({len(train_df_place)}) for initial sequence.")
             continue
        initial_input = train_df_place[input_features].values[-seq_length:]

        y_pred_list = [] # Store raw predictions for this place

        with torch.no_grad():
            current_input_seq = initial_input.copy() # Start with raw training data tail

            # Iterate through the test set to make predictions step-by-step
            if len(X_test) == 0:
                print(f"No test data points available for {place} after processing.")
                continue

            for i in range(len(X_test)):
                 # Prepare input tensor (b=1, t=seq_len, f=enc_in)
                 seq_tensor = torch.FloatTensor(current_input_seq).unsqueeze(0).to(device)

                 # Predict `pred_length` steps ahead (raw scale)
                 prediction = model(seq_tensor, None, None, None) # Shape (1, pred_len, c_out)

                 # Take only the *first* step prediction from the `pred_length` output
                 first_step_pred = prediction[0, 0, :].cpu().numpy() # Shape (c_out,) - raw scale

                 # Store this first step prediction
                 y_pred_list.append(first_step_pred)

                 if i < len(X_test): # Ensure we don't go out of bounds
                     actual_next_full_features = X_test[i, :] # Shape (enc_in,) - raw scale
                 else:
                     # This case shouldn't be reached with the loop range, but as a safeguard
                     print("Warning: Reached end of X_test unexpectedly during prediction loop.")
                     break

                 # Update the input sequence: drop oldest, append actual next step's features
                 current_input_seq = np.vstack((current_input_seq[1:], actual_next_full_features))

        y_pred = np.array(y_pred_list) # Shape (n_test_steps, c_out)

        num_predictions = len(y_pred)
        if num_predictions == 0:
             print(f"No predictions were generated for {place}. Skipping metrics.")
             continue

        # Extract the target features from the raw test data
        y_test = X_test[:num_predictions, feature_indices] # Extract only target features, align length

        # Align test dates with the number of predictions made
        aligned_test_dates = test_dates[:num_predictions]

        # Store results
        # eval_dict[place] = {
        #     "num pred": num_predictions
        # }

        # --- Calculate and Print Metrics (No MAPE) ---
        print(f"{place} Metrics:")
        metrics_summary = {}
        for i, feature in enumerate(elements_list):
            # Ensure y_test and y_pred have the same length for metrics
            if len(y_test[:, i]) != len(y_pred[:, i]):
                print(f"Warning: Length mismatch for feature '{feature}' in {place}. Actual: {len(y_test[:, i])}, Pred: {len(y_pred[:, i])}. Skipping metrics.")
                continue

            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])

            se_list = np.sqrt((y_test[:, i]-y_pred[:, i])**2)
            max_se = np.max(se_list) if len(se_list) > 0 else 0 # Handle empty case

            # Updated metrics dictionary (no mape)
            metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "HRSE":max_se}
            eval_dict[feature] = metrics
            metrics_summary[feature] = metrics
            # Updated print statement (no mape)
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
    model_output_length: int,
    num_lags: int = 12,
):
    # Config
    device = 'cpu'
    
    print(f"\n--- Predicting {n_steps} steps ahead for Place={place_to_test}, Elements={element_column} ---")

    required_cols = ['date'] + element_column
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame missing required columns: {required_cols}")
        return None

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    # Extract and prepare data
    df_location = df[df[place_column_name] == place_to_test].copy()
    df_location['date'] = pd.to_datetime(df_location['date'])
    #df_location.sort_values(by='date')
    df_location.set_index('date', inplace=True)

    for col in element_column:
        df_location[col] = df_location[col].interpolate(method='linear')

    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Prepare prediction structures
    df_predict_recursive = df_location.copy()
    current_last_date = df_predict_recursive.index[-1]
    
    # Build initial input (last known num_lags steps)
    recent_data = df_predict_recursive[element_column].values[-num_lags:]
    recent_dates = df_predict_recursive.index[-num_lags:]

    initial_input = []
    for i in range(num_lags):
        date = recent_dates[i]
        time_feats = [date.day, date.month, date.year]
        input_vector = np.concatenate([recent_data[i], time_feats])
        initial_input.append(np.array(input_vector).flatten())  # Ensure 1D

    current_input_seq = np.vstack(initial_input)  # Shape: (num_lags, enc_in)
    # raise Exception(current_input_seq)
    # current_input_seq = np.array(initial_input)
    y_pred = []
    prediction_output = {col: [] for col in element_column}

    with torch.no_grad():
        for step in range(0, n_steps, model_output_length):
            # Predict future steps
            seq_tensor = torch.FloatTensor(current_input_seq).unsqueeze(0).to(device)
            prediction = model(seq_tensor, None, None, None).cpu().numpy()  # (1, pred_len, c_out)

            current_pred_len = min(model_output_length, n_steps - len(y_pred))
            pred_chunk = prediction[0,:current_pred_len]
            y_pred.extend(pred_chunk)
                    
            # Update input sequence with predicted values + time features
            for j in range(current_pred_len):
                current_last_date += timedelta(days=freq_days)
                time_feats = [current_last_date.day, current_last_date.month, current_last_date.year]
                pred_values = np.array(pred_chunk[j])
                pred_values[-3:] = time_feats  # replace last 3 values
                for i in range (len(element_column)):
                    prediction_output[element_column[i]].append({
                        'predicted_date': current_last_date,
                        'predicted_value': pred_values[i]
                    })
                current_input_seq = np.vstack((current_input_seq[1:], pred_values))
                
    print("\n--- Multi-step prediction complete ---")
    return prediction_output