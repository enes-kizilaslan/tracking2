import os

MODEL_DIR = 'models'
MODEL_LIST = [f[:-4] for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
FEATURE_FILE = "selected_features.xlsx"
PERFORMANCE_FILE = "model_performance.xlsx" 
