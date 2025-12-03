# Media Framing Classification & Analysis

**Milestone-3---Team-22---3M**  

## Project Overview

Our project tracks how media topic framing shifts over time across U.S. outlets with different political leanings (left, right, center). We will fine tune a BERT transformer model using the Multi-Modal Framing dataset to classify news articles into an established frame categories. We will use the results to quantify drift across political leanings and over time.


## Required Libraries

Install the following Python packages:

```
pip install -r requirements.txt
```

##  Model Information
- Base model: microsoft/deberta-v3-base 
- Fine-tuned on: MMF framing dataset with custom threshold tuning using macro-F1 optimization
- Supports multi-label classification

##  Data Sources
- Multi-Modal Framing (MMF) Dataset (https://arxiv.org/html/2503.20960v1#S3)
 - This serves as our primary supervised dataset for training and evaluating our frame classification models.
- U.S. News Media Coverage & Outlet Bias Database (https://zenodo.org/records/7476697)
 - This serves as our dataset for downstream analysis of framing patterns over time and across outlets/
- All Sides Rankings (https://www.kaggle.com/datasets/supratimhaldar/allsides-ratings-of-bias-in-electronic-media)
 - This serves as our dataset to determine political leaning for various outlets.


## Team Members
- **Mark Griffin**
  - Zero Shot Topic Clasification
  - Gold Standard Labeling
  - Topic_Classification.ipynb, Data_Merge_and_EDA.ipynb, Event_Framing_Analysis.ipynb,Frame_Cooccurrence_Analysis.ipynb,Framing_Drift_Analysis.ipynb,Outlet_Framing_Analysis.ipynb,Key_Findings.ipynb
- **Matt Cott** 
  - Gold Standard Subset Creation
  - Gold Standard Labeling
  - Gold_Standard_Set_Up.ipynb, Topic Analysis_MC.ipynb
- **Michael Giarmarco** 
  - Data Collection/Cleaining
  - DeBERTa-v3 Training, Evaluation, and Classification
  - Gold Standard Labeling and Evaluation
  - helpers.py, Train_MMF.ipynb, Gold_Standard_Eval.ipynb, Data_Collection.ipynb, Data_Collection_New_Sources.ipynb, Framing.ipynb, Framing_New_Sources.ipynb, Frame_Distribution_Cleaning.ipynb, Mickey_Vis_Analysis.ipynb



  
