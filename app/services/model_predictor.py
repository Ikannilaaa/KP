import cloudpickle
import pandas as pd
import json
import os

# Load model saat modul diimpor
MODEL_PATH = os.path.abspath("./app/models/posture_model.pkl")

# with open(MODEL_PATH, "rb") as f:
#     model = cloudpickle.load(f)

model = None
try:
    with open(MODEL_PATH, "rb") as f:
        model = cloudpickle.load(f)
except Exception as e:
    print(f"⚠️  WARNING: Gagal load model ML ({e}). Menggunakan mode DUMMY.")
    model = None

def convert_to_python_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(i) for i in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

def predict_from_keypoints_df(keypoint_df):
    if model is None:
        return {
            "rula_score": 0,
            "reba_score": 0,
            "label_rula": "Aman (Dummy)",
            "label_reba": "Aman (Dummy)",
            "feedback": "Model ML tidak aktif."
        }
    
    dataframe = pd.DataFrame([keypoint_df])
    result = model.predict_from_keypoints(dataframe)
    result_dict = convert_to_python_type(result)
    return result_dict