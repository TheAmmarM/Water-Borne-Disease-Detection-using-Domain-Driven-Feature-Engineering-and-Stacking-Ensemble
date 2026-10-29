# ==================================================
# Phase 0: Import Libraries & Load Data
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
df = pd.read_excel('WQD.xlsx')

# Display basic info
print("=== DATASET SHAPE ===")
print(f"Shape: {df.shape}")
print("\n=== FIRST 5 ROWS ===")
print(df.head())
print("\n=== COLUMN NAMES & DATA TYPES ===")
print(df.dtypes)