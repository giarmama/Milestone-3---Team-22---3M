import torch
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42
np.set_printoptions(precision=5, suppress=True)


# Globals
tok = None
model = None
frames = None
best_thresh = None


def Load_model(save_dir: str):
    """
    Load tokenizer, model, frame list, and best threshold from a directory.
    Call this once before using Predict_frames().
    """
    global tok, model, frames, best_thresh

    tok = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)

    with open(f"{save_dir}/frames.json") as f:
        frames = json.load(f)

    with open(f"{save_dir}/threshold.json") as f:
        best_thresh = json.load(f)["global"]

    print(f"Model, tokenizer, frames, and threshold loaded from: {save_dir}")
    return model, tok, frames, best_thresh


def Predict(article: str):
    """Predict frames and return labels, vectors, and probabilites."""


    # Tokenize
    inputs = tok(
        article,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    probs = probs.round(5)
    preds = (probs >= best_thresh).astype(int)

    frames = Vec_to_frame(preds)

    return frames, preds, probs

def Vec_to_frame(vec):
    return[f for f,v in zip(frames,vec) if v ==1]

def Row_to_vector(row):
    """ Convert gold standard row to a vector."""
    FRAME_COL_MAP = {
    "economic": "L_economics",
    "fairness": "L_fairness",
    "public_op": "L_public opinion",
    "political": "L_political",
    "quality_life": "L_QOL",
    "crime": "L_crime",
    "culture": "L_culture",
    "health": "L_health",
    "legality": "L_legality",
    "morality": "L_morality",
    "policy": "L_policy",
    "regulation": "L_regulation",
    "security": "L_security",
    "cap&res": "L_capacity_resources",
    }
    
    vec = []
    for f in frames:
        col = FRAME_COL_MAP[f]
        val = row.get(col, np.nan)
        vec.append(0 if pd.isna(val) else int(val > 0))
    return np.array(vec, dtype=int)

def Eval_against_gold(gold_df,vector_col=None):
    y_true = []
    y_pred = []
    
    for i, row in gold_df.iterrows():
        text = row["content"]
        
        if vector_col:
            gold_vec = np.array(row[vector_col], dtype=int)
        else:            
            gold_vec = Row_to_vector(row)
        pred_vec, _ = Predict_vector(text)
    
        y_true.append(gold_vec)
        y_pred.append(pred_vec)
    
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    
    print("Evaluated articles:", y_true.shape[0])
    return y_true,  y_pred

def Is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False