import DLModels.Config
from DLCode import *
import logging

# Logging config
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for verbose logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from MLCodeForAPI import preprocessData
from MLCodeForAPI import readCSVfile

def trainWithDLModel(df, elements_list, date_column_name,place_column_name, place_id, date_tag, train_test_ratio= 0.7, model_dir='saved_models'):
    logger.info(f"Starting training for place {place_id} with date tag {date_tag}")
    # df = readCSVfile (file)

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
    column_list = elements_list + ['day','month','year']
    
    #import models from DLModels
    import sys
    #config for general DL Models:
    LAG = 12
    d_ff = 1024
    dropout = 0.1
    
    #config for each model
    from DLModels.Config import DictConfigModels,Config
    config = Config (seq_len=LAG,pred_len=4,enc_in=len(column_list),c_out=len(column_list),d_ff=d_ff,dropout=dropout)
    configList = DictConfigModels(config)
    
    from DLModels.ETSFormer import ETSformer
    from DLModels.ETSFormerPar import ETSformerPar
    model_type_list = [
        # ("ETSformer", ETSformer(configList.ETSFormerConfig)),
        ("ETSformerPar", ETSformerPar,configList.ETSFormerConfig),
        ("ETSformer", ETSformer,configList.ETSFormerConfig)
    ]
    
    model_result = {}
    eval_result = {}
    
    for model_type_name,model_class,config in model_type_list:
        logger.info(f"Training {model_type_name} model...")

        model_path,eval_dict = \
            train_export_model_DL_SingleLocation(df = df, 
                                                elements_list= elements_list, 
                                                PLACE_TO_TEST = place_id,
                                                place_column_name = place_column_name, 
                                                config = config,
                                                date_tag=date_tag,
                                                train_split_ratio=train_test_ratio, 
                                                base_model_class=model_class, 
                                                base_model_name=model_type_name, 
                                                model_dir=model_dir)
        if model_path:
            model_result[model_type_name] = model_path
            eval_result[model_type_name] = eval_dict
            
            logger.info(f"Model saved to {model_path}")
        else:
            logger.warning(f"No model path returned for {model_type_name}")

    return model_result,eval_result

def predictWithDLModel(df, num_step, freq_days, elements_list, date_column_name,place_column_name, place_id, date_tag, model_dir="saved_models"):
    logger.info(f"Starting prediction for place {place_id} with model date tag {date_tag}")

    df = preprocessData(df, elements_list,place_column_name, date_column_name)

    #import models from DLModels
    import sys
    #config for general DL Models:
    LAG = 12
    d_ff = 1024
    dropout = 0.1
    
    #config for each model
    from DLModels.Config import DictConfigModels,Config
    config = Config (seq_len=LAG,pred_len=4,enc_in=len(elements_list)+3,c_out=len(elements_list)+3,d_ff=d_ff,dropout=dropout)
    configList = DictConfigModels(config)
    
    from DLModels.ETSFormer import ETSformer
    from DLModels.ETSFormerPar import ETSformerPar
    model_type_list = [
        # ("ETSformer", ETSformer(configList.ETSFormerConfig)),
        ("ETSformerPar", ETSformerPar,configList.ETSFormerConfig),
        ("ETSformer", ETSformer,configList.ETSFormerConfig)
    ]

    model_result = {}
    for model_type_name,model_class,config in model_type_list:
        filename = f"{model_type_name}_multitarget_place{place_id}_{date_tag}.pkl"
        model_path = os.path.join(model_dir, filename)

        if not os.path.exists(model_path):
            logger.warning(f"Model {model_type_name} not found at {model_path}, skipping...")
            continue 

        logger.info(f"Running prediction with {model_type_name} model...")
        results = predict_future_steps_DL(
            df=df,
            freq_days=freq_days,
            place_to_test=place_id,
            place_column_name=place_column_name,
            element_column=elements_list,
            n_steps=num_step,
            num_lags=12,
            model_output_length = config.enc_in,
            model_path=model_path
        )
        # raise Exception (type(results))

        if results:
            model_result[model_type_name] = results
            logger.info(f"Prediction successful with {model_type_name}")
        else:
            logger.warning(f"No results for model type: {model_type_name}")

    
    return model_result