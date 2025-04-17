from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from MLCodeForAPI import trainWithMLModel, predictWithMLModel
import json
import datetime
import numpy as np

import logging

# Logging config
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for verbose logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def convert_np(obj):
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

app = FastAPI()
import sys
print (sys.version)

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictParams(BaseModel):
    elements_list: List[str]
    date_column_name: str
    place_id: int
    date_tag: str
    num_step: int = 5
    freq_days: int = 1
    model_dir: Optional[str] = "saved_models"

class TrainParams(BaseModel):
    elements_list: List[str]
    date_column_name: str
    place_id: int
    date_tag: str
    model_dir: Optional[str] = "saved_models"


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    elements_list: List[str] = Form(...),
    date_column_name: str = Form(...),
    place_id: int = Form(...),
    date_tag: str = Form(...),
    num_step: int = Form(4),
    freq_days: int = Form(7),
    model_dir: Optional[str] = Form("saved_models")
):
    try:
        parsed_elements_list = [e.strip() for e in elements_list[0].split(",") if e.strip()]
        result = predictWithMLModel(
            file=file,
            num_step=num_step,
            freq_days=freq_days,
            elements_list=parsed_elements_list,
            date_column_name=date_column_name,
            place_id=place_id,
            date_tag=date_tag,
            model_dir=model_dir
        )
        return {"success": True, "results": convert_np(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/train")
async def train(
    file: UploadFile = File(...),
    elements_list: List[str] = Form(...),  
    date_column_name: str = Form("Ngày quan trắc"),
    train_test_ratio: float = Form(0.7),
    place_id: int = Form(1),
    date_tag: str = Form("170425"),
    model_dir: Optional[str] = Form("saved_models")
):
    """_summary_

    Args:\n
        file (UploadFile, optional): Upload file csv of data.
        elements_list (List[str], optional): List of elements to train model, keep list for prediction.
        date_column_name (str, optional): name of the "date" column, default "Ngày quan trắc".
        train_test_ratio (float, optional): ratio to split train/test from 0.1->1, default 0.7, 1 does not return model test result.
        place_id (int, optional): Place to train model Defaults to Form(1).
        date_tag (str, optional): Date tag to keep track of model, recommend ddmmyy form. Default 170425.
        model_dir (Optional[str], optional): Folder to save models in server. Defaults Form("saved_models").

    Returns:
        _type_: _description_
    """
    try:
        parsed_elements_list = [e.strip() for e in elements_list[0].split(",") if e.strip()]
        model_path,eval_dict = trainWithMLModel(
            file=file,
            elements_list=parsed_elements_list,
            date_column_name=date_column_name,
            train_test_ratio=train_test_ratio,
            place_id=place_id,
            date_tag=date_tag,
            model_dir=model_dir
        )
        return {"success": True, "model_paths": model_path, "evaluation":eval_dict}
    except Exception as e:
        return {"success": False, "error": str(e)}