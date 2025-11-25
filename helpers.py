import torch
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, DetectorFactory
from tqdm import tqdm
tqdm.pandas()
DetectorFactory.seed = 42
np.set_printoptions(precision=5, suppress=True)


# Globals
tok = None
model = None
frames = None
best_thresh = None
device = None


def Load_model(save_dir: str):
    """
    Load tokenizer, model, frame list, and best threshold from a directory.
    Call this once before using Predict_frames().
    """
    global tok, model, frames, best_thresh, device

    tok = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)

    with open(f"{save_dir}/frames.json") as f:
        frames = json.load(f)

    with open(f"{save_dir}/threshold.json") as f:
        best_thresh = json.load(f)["global"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model, tokenizer, frames, and threshold loaded from: {save_dir}")
    print(f"Using device: {device}")
    return model, tok, frames, best_thresh


def Predict(texts, batch_size=32):
    """
   Predict frames and probabilities in batches.
    """
    device = next(model.parameters()).device
    
    all_frames = []
    all_vecs = []
    all_probs = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]

        # Tokenize whole batch at once
        inputs = tok(
            batch,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()

        probs = probs.round(5)
        preds = (probs >= best_thresh).astype(int)

        # Convert vectors to frame lists
        for vec in preds:
            all_frames.append(Vec_to_frame(vec))

        all_vecs.append(preds)
        all_probs.append(probs)

    all_vecs = np.vstack(all_vecs)
    all_probs = np.vstack(all_probs)
    return all_frames, all_vecs, all_probs


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
        _, pred_vec, _ = Predict([text])
        pred_vec = pred_vec[0]
    
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