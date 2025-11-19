import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
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


def Predict_frames(article: str, return_all: bool = True):
    """Return a sorted dataframe OR a list of predicted frames."""


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

    # Build dataframe
    results = [
        {"frame": f, "prob": float(p), "predicted": bool(pred)}
        for f, p, pred in zip(frames, probs, preds)
    ]

    df = pd.DataFrame(results)
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)

    if return_all:
        return df
    else:
        return df[df["predicted"] == True]["frame"].tolist()

def Predict_vector(article: str):
    """Return (preds, probs) for a single article."""
    inputs = tok(
        article,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    probs = probs.round(5)

    preds = (probs >= best_thresh).astype(int)
    return preds, probs