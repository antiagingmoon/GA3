# backend.py

import pandas as pd
import random

def load_patient_data(expression_file, metadata_file):
    expr_df = pd.read_excel(expression_file, index_col=0)
    meta_df = pd.read_excel(metadata_file)
    return expr_df, meta_df

def predict_cancer_type(expr_df, patient_id):
    # 예측 대신 무작위로 암 타입 반환 (데모용)
    return random.choice(["Lung", "Breast", "Colon", "Prostate"])
