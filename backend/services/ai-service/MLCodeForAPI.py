from MLCode import *
import logging

# Logging config
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for verbose logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def preprocessData(data, elements_list,place_column_name, date_column_name):
    logger.info("Preprocessing data...")
    data.ffill()
    
    if date_column_name not in data.columns:
        raise Exception(f"[ERROR] '{date_column_name}' not found in columns!")
    
    for element in elements_list:
        if element not in data.columns:
            raise Exception(f"[ERROR] '{element}' not found in columns!")
        data[element] = data[element].replace({'KPH': 0})
    
        
    data[elements_list] = data[elements_list].replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce')
    data[date_column_name] = pd.to_datetime(data[date_column_name], errors='coerce')
    
    data['date'] = pd.to_datetime(data[date_column_name], dayfirst=True)
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data = data.sort_values(by='date')
    
    data = data.dropna(subset=elements_list)
    # Minh troll
    # data = data[(data['year'] == 2022) | (data['year'] == 2023)]
    
    columnToKeep = elements_list + [place_column_name]+ ['date','day','month','year']
    available_columns = [col for col in columnToKeep if col in data.columns]
    data = data[available_columns]
    logger.info("Preprocessing complete.")
    return data

def predictWithMLModel(file, num_step, freq_days, elements_list, date_column_name,place_column_name, place_id, date_tag, model_dir="saved_models"):
    logger.info(f"Starting prediction for place {place_id} with model date tag {date_tag}")
    try:
        if isinstance(file, str):
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(file.file)
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")

    if elements_list is None:
        logger.error("Element list is None, unable to determine the elements")
        return None
    if date_column_name is None:
        logger.error("Date column name is None, unable to determine the date column name")
        return None
    if date_tag is None:
        logger.error("Date tag is None, unable to determine the date tag")
        return None

    df = preprocessData(df, elements_list,place_column_name, date_column_name)

    model_type_list = ['rf', 'xgb']
    model_result = {}
    for model_type_name in model_type_list:
        filename = f"{model_type_name}_multitarget_place{place_id}_{date_tag}.pkl"
        model_path = os.path.join(model_dir, filename)

        if not os.path.exists(model_path):
            logger.warning(f"Model {model_type_name} not found at {model_path}, skipping...")
            continue

        logger.info(f"Running prediction with {model_type_name} model...")
        results = predict_future_steps(
            df=df,
            freq_days=freq_days,
            place_to_test=place_id,
            place_column_name=place_column_name,
            element_column=elements_list,
            n_steps=num_step,
            num_lags=12,
            model_path=model_path
        )
        # raise Exception (type(results))

        if results:
            model_result[model_type_name] = results
            logger.info(f"Prediction successful with {model_type_name}")
        else:
            logger.warning(f"No results for model type: {model_type_name}")

    
    return model_result

def trainWithMLModel(file, elements_list, date_column_name,place_column_name, place_id, date_tag, train_test_ratio= 0.7, model_dir='saved_models'):
    NUM_LAGS = 12
    logger.info(f"Starting training for place {place_id} with date tag {date_tag}")
    try:
        if isinstance(file, str):
            df = pd.read_csv(file,encoding='utf-8-sig')
        else:
            df = pd.read_csv(file.file,encoding='utf-8-sig')
        
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise Exception (f"Error reading CSV: {e}")
        # return None

    if elements_list is None:
        logger.error("Element list is None, unable to determine the elements")
        raise Exception (f"Element list is None, unable to determine the elements:")
    if date_column_name is None:
        logger.error("Date column name is None, unable to determine the date column name")
    if date_tag is None:
        logger.error("Date tag is None, unable to determine the date tag")
        return None
    
    df.columns = df.columns.str.strip()
    df = preprocessData(df, elements_list,place_column_name, date_column_name)
    os.makedirs(model_dir, exist_ok=True)

    model_type_list = ["rf", "xgb"]
    model_result = {}
    eval_result = {}
    
    for model_type_name in model_type_list:
        logger.info(f"Training {model_type_name} model...")
        if model_type_name == "rf":
            base_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        elif model_type_name == "xgb":
            base_model = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1)

        model_path,eval_dict = train_export_model(df, elements_list, place_id,place_column_name, NUM_LAGS, train_split_ratio=train_test_ratio, base_model=base_model, base_model_name=model_type_name, model_dir=model_dir)
        if model_path:
            model_result[model_type_name] = model_path
            eval_result[model_type_name] = eval_dict
            
            logger.info(f"Model saved to {model_path}")
        else:
            logger.warning(f"No model path returned for {model_type_name}")

    return model_result, eval_result, model_type_list
